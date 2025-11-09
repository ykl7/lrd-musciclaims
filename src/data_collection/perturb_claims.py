"""This script perturbs the SUPPORT claims into CONTRADICT claims using LLM via vLLM server.
if the JUDGE step is enabled, it also verifies the generated contradictions (code to change is "is_valid = ??").
It processes claims asynchronously with controlled concurrency to optimize throughput."""

import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm
import asyncio
import re

# --- Configuration & Setup ---
try:
    # Use AsyncOpenAI instead of OpenAI
    client = AsyncOpenAI(
        base_url="http://localhost:8002/v1",
        api_key="EMPTY"
    )
    # Note: We'll validate connection in the async main function
    print("✅ Async client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize async client.")
    print(f"   Error: {e}")
    exit()

async def validate_connection():
    """Validate the connection to the model server."""
    try:
        await client.models.list()
        print("✅ Successfully connected to the Qwen model via local tunnel.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to the model. Is your SSH tunnel running or is the vLLM server down?")
        print(f"   Error: {e}")
        return False

async def generate_contradiction_with_reflection(claim_text: str) -> str:
    """
    STEP 1: The GENERATOR. Its job is to create a candidate contradiction.
    Now async!
    """
    prompt = f"""
You are an expert in scientific writing and logic. Your task is to act as a minimalist editor. You will first provide your reasoning and then rewrite the scientific claim to be a direct contradiction.

Follow these rules:
1.  **Reasoning First:** Briefly explain the core assertion and your strategy for contradicting it.
2.  **Factually Opposite:** The new claim must be factually opposite to the original.
3.  **Maintain Style:** Maintain a similar formal, scientific style and length.
4.  **No Simple Negation:** Do not simply add "not". Rephrase the core assertion.
5.  **Plausible:** Ensure the contradiction is plausible within a scientific context.
6.  **PRESERVE VALUES (CRITICAL RULE):** Do NOT invent new numbers, percentages, or data values. You must preserve all original numerical data. Change the *relationship* between them (e.g., "increased" to "decreased", "more than" to "less than"), not the values themselves.

A few examples:

Original Claim: "The proportion of funded research has fluctuated around 50% since 2011."
Reasoning: The original claim describes fluctuation. The contradiction will assert the opposite: stability. Crucially, I will keep the original value of "50%" as the stable reference point.
Contradictory Claim: "The proportion of funded research has remained stable at 50% since 2011."

---

Original Claim: "This effect was only observed in patients over the age of 60."
Reasoning: The original claim limits the effect to a specific group ("over 60"). The contradiction will broaden the effect to all groups, preserving the "60" as a boundary.
Contradictory Claim: "This effect was observed across all patient age groups, including those under 60."

---

Original Claim: "Accordingly, the number of citable articles has increased steadily year-on-year."
Reasoning: The original claim states a clear upward trend. The contradiction will state the opposite trend (no increase or a decrease) without inventing any new numbers.
Contradictory Claim: "Accordingly, the number of citable articles has remained flat or decreased in recent years."

---
### YOUR TASK
---

Original Claim: "{claim_text}"
Reasoning:
"""
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            stop=["---"]
        )
        full_response_text = response.choices[0].message.content.strip()
        
        potential_claim = ""
        if "Contradictory Claim:" in full_response_text:
            potential_claim = full_response_text.split("Contradictory Claim:")[-1]
        else:
            lines = full_response_text.split('\n')
            for line in reversed(lines):
                if line.strip():
                    potential_claim = line
                    break
        
        final_claim = potential_claim.strip().strip('"').strip("'").strip()
        return final_claim if final_claim else full_response_text

    except Exception as e:
        print(f"\nError during GENERATOR call for claim '{claim_text[:50]}...': {e}")
        return None

async def is_contradiction(original_claim: str, perturbed_claim: str) -> bool:
    """
    STEP 2: The JUDGE. Its only job is to verify if the pair is a true contradiction.
    Now async!
    """
    prompt = f"""
You are a logical validation expert. You are given an original claim and a perturbed claim. Your task is to judge whether the perturbation is a valid contradiction. A valid contradiction means both claims cannot be true at the same time.

- If they are a valid contradiction, respond with only the word "Yes".
- If they are NOT a valid contradiction (e.g., the meaning is the same, it's irrelevant, or not a logical opposite), respond with only the word "No".

---
Original Claim: "{original_claim}"
Perturbed Claim: "{perturbed_claim}"
---

Judgment:
"""
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        judgment = response.choices[0].message.content.strip().lower()
        return "yes" in judgment

    except Exception as e:
        print(f"\nError during JUDGE call for claim '{original_claim[:50]}...': {e}")
        return False

async def process_single_claim(index: int, row: pd.Series, pbar: tqdm) -> dict:
    """
    Process a single claim: generate contradiction and judge it.
    Returns the new row dict if valid, None otherwise.
    """
    original_claim = row['claim']
    
    # Step 1: Generate a candidate
    perturbed_claim = await generate_contradiction_with_reflection(original_claim)
    
    if not perturbed_claim:
        print(f"\nSkipping pair #{index + 1} due to generation failure.")
        pbar.update(1)
        return None
        
    # Step 2: Judge the candidate
    # is_valid = await is_contradiction(original_claim, perturbed_claim)
    is_valid = True
    
    # --- Print the full comparison with the judgment ---
    print("\n" + "="*80)
    print(f"PAIR #{index + 1}")
    print(f"  Original:    {original_claim}")
    print(f"  Perturbed:   {perturbed_claim}")
    print(f"  Judgment:    {'VALID' if is_valid else 'INVALID'}")
    print("="*80)
    
    pbar.update(1)
    
    # Only return the row if the Judge approved it
    if is_valid:
        new_row = row.to_dict()
        new_row['claim'] = perturbed_claim
        new_row['class'] = 'CONTRADICT'
        return new_row
    
    return None

async def perturb_dataframe_claims(df: pd.DataFrame, max_concurrent: int = 5) -> pd.DataFrame:
    """
    Process claims asynchronously with controlled concurrency.
    
    Args:
        df: Input DataFrame
        max_concurrent: Maximum number of concurrent API calls (adjust based on your server capacity)
    """
    support_df = df[df['class'] == 'SUPPORT'].copy()
    
    print(f"\nGenerating and Judging contradictions for {len(support_df)} support claims...")
    print(f"Running with max {max_concurrent} concurrent requests...")
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(index, row, pbar):
        async with semaphore:
            return await process_single_claim(index, row, pbar)
    
    # Create progress bar
    pbar = tqdm(total=len(support_df), desc="Processing claims")
    
    # Create all tasks
    tasks = [
        process_with_semaphore(index, row, pbar)
        for index, row in support_df.iterrows()
    ]
    
    # Run all tasks concurrently (but limited by semaphore)
    results = await asyncio.gather(*tasks)
    
    pbar.close()
    
    # Filter out None results
    new_rows = [row for row in results if row is not None]

    if not new_rows:
        print("Warning: No new valid claims were generated after judging.")
        return df

    perturbed_df = pd.DataFrame(new_rows)
    final_df = perturbed_df.sort_values(by=['figure_id', 'class']).reset_index(drop=True)
    
    return final_df

async def main():
    """Main async function."""
    # Validate connection
    if not await validate_connection():
        exit()
    
    # Load data
    support_data_path = 'all_support_data.csv'
    df_initial = pd.read_csv(support_data_path)
    
    # Process claims
    df_final = await perturb_dataframe_claims(df_initial, max_concurrent=10)
    
    print("\n--- Final DataFrame with VALIDATED Perturbations ---")
    print(df_final)
    
    # Save results
    output_filename = "all_perturbed_data_without_judge.csv"
    df_final.to_csv(output_filename, index=False)
    print(f"\n✅ Final, high-quality data saved to '{output_filename}'")

# --- Main Execution ---
if __name__ == "__main__":
    asyncio.run(main())
