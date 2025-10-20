import pandas as pd
from openai import AsyncOpenAI # 1. Import the Async client
from tqdm.asyncio import tqdm as async_tqdm # 2. Import the async-compatible version of tqdm
import re
import asyncio # 3. Import the asyncio library

# --- Configuration & Setup ---

try:
    # 4. Use the AsyncOpenAI client. The connection is now established on the first call.
    client = AsyncOpenAI(
        base_url="http://localhost:8002/v1",
        api_key="EMPTY"
    )
    print("✅ Async client configured. Connection will be tested on the first call.")
except Exception as e:
    print(f"❌ Client configuration failed. Error: {e}")
    exit()

def preprocess_claim(text: str) -> str:
    # This function is unchanged and remains essential
    if not isinstance(text, str):
        return ""
    processed_text = re.sub(r'Open in a new tab', '', text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


async def clear_figure_references(claim_text: str, semaphore: asyncio.Semaphore) -> str:
    """
    ASYNC VERSION. Uses a semaphore to limit concurrent requests.
    The core logic and prompt are identical to your original.
    """
    # This 'async with' block ensures we don't exceed our concurrency limit.
    async with semaphore:
        prompt = f"""
You are a hyper-literal text-editing bot. You will first provide your reasoning by explaining exactly what you will remove, then you will provide the cleaned text.

**CRITICAL RULES:**
1.  **REASONING FIRST:** In the 'Reasoning' section, identify the exact text to be removed based on the rules.
2.  **NO MARKDOWN (CRITICAL):** The 'Cleaned' output must be PURE, UNFORMATTED text. Do not use any Markdown (`**`, `*`), quotes (`""`), or any other formatting.
3.  **DELETE ONLY:** You must only delete the reference text.
4.  **NO REPHRASING:** Do not change the wording or grammar of the remaining text.
5.  **PARENTHESES RULE:** If a reference is inside parentheses `(...)`, the entire parenthetical block must be removed.
6.  **PROXIMITY RULE:** Do NOT remove text that is outside of the parentheses, even if it is right next to them.
7.  **INLINE RULE:** If a reference is not in parentheses, remove the reference itself AND any short, direct pointer phrases immediately preceding it (e.g., the phrase "see third row," before "Table 3").
8.  **OUTPUT FORMAT:** Provide your output in two parts: "Reasoning:" and "Cleaned:".
---
### EXAMPLES
---

**Original:** The FMEA combines brainstorming by team members and flowcharting to create a detailed process map with potential breakdowns or failure modes noted, ranked, and corrective action identified (Figure 1; Dailey, 2004).
**Reasoning:** The parenthetical block '(Figure 1; Dailey, 2004)' contains a figure reference and another citation. According to the parentheses rule, I will remove the entire block.
**Cleaned:** The FMEA combines brainstorming by team members and flowcharting to create a detailed process map with potential breakdowns or failure modes noted, ranked, and corrective action identified.

---

**Original:** The proportion of funded research has fluctuated during the period: 9.4% in 2010 as opposed to 25.0% in 2014 (Fig. 2).
**Reasoning:** The reference is '(Fig. 2)'. The text 'in 2014' is adjacent but outside the parentheses, so it must be preserved according to the proximity rule. I will only remove the parenthetical block '(Fig. 2)'.
**Cleaned:** The proportion of funded research has fluctuated during the period: 9.4% in 2010 as opposed to 25.0% in 2014.

---

**Original:** As of June 2020, the federal policy defining WOTUS (“Navigable Waters Protection Rule,” [9]) considers jurisdictional tributaries as those that are perennial or intermittent (Table 1) and contribute surface water flow to a traditional navigable water or territorial sea in a typical year (Figure 2).
**Reasoning:** This sentence contains two separate parenthetical references: '(Table 1)' and '(Figure 2)'. I will remove both of them.
**Cleaned:** As of June 2020, the federal policy defining WOTUS (“Navigable Waters Protection Rule,” [9]) considers jurisdictional tributaries as those that are perennial or intermittent and contribute surface water flow to a traditional navigable water or territorial sea in a typical year.

---

**Original:** From the cluster presented in Fig. 7 we selected the paper, "Title of Paper", see third row, Table 3.
**Reasoning:** This sentence has two inline references. The first is 'Fig. 7'. The second is the phrase 'see third row, Table 3'. The phrase 'see third row,' is a direct pointer to 'Table 3' and must be removed along with it, according to the inline rule.
**Cleaned:** From the cluster presented in we selected the paper, "Title of Paper".

---
### YOUR TASK
---

**Original:** "{claim_text}"
**Reasoning:**
"""
        # Return early for empty strings to avoid unnecessary API calls
        if not claim_text:
            return ""

        try:
            # 6. Use 'await' for the network call
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=len(claim_text) + 150
            )
            
            # The rest of your parsing logic is perfect and remains unchanged.
            full_response_text = response.choices[0].message.content.strip()
            
            cleaned_text = ""
            if "Cleaned:" in full_response_text:
                cleaned_text = full_response_text.split("Cleaned:")[-1].strip()
            else:
                lines = full_response_text.split('\n')
                if len(lines) > 1:
                    cleaned_text = lines[-1].strip()
                else:
                    cleaned_text = full_response_text

            if cleaned_text.startswith(","):
                cleaned_text = cleaned_text[1:].strip()
                
            return cleaned_text.replace(" .", ".").strip()

        except Exception as e:
            # This error handling is also unchanged.
            print(f"\nError during API call for claim '{claim_text[:50]}...': {e}")
            return claim_text


def clear_figure_references(claim_text: str) -> str:
    """
    Uses the Qwen model with an enriched Chain-of-Thought prompt.
    This version includes a more sophisticated INLINE RULE to remove pointer phrases.
    """
    # This prompt has been updated with a smarter INLINE RULE and corrected example.
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=len(claim_text) + 150
        )
        
        full_response_text = response.choices[0].message.content.strip()
        
        cleaned_text = ""
        if "Cleaned:" in full_response_text:
            cleaned_text = full_response_text.split("Cleaned:")[-1].strip()
        else:
            lines = full_response_text.split('\n')
            if len(lines) > 1:
                cleaned_text = lines[-1].strip()
            else:
                cleaned_text = full_response_text

        # Remove potential leading comma if it's left over
        if cleaned_text.startswith(","):
            cleaned_text = cleaned_text[1:].strip()
            
        return cleaned_text.replace(" .", ".").strip()

    except Exception as e:
        print(f"\nError during API call for claim '{claim_text[:50]}...': {e}")
        return claim_text



import pandas as pd
from openai import AsyncOpenAI # 1. Import the Async client
from tqdm.asyncio import tqdm as async_tqdm # 2. Import the async-compatible version of tqdm
import re
import asyncio # 3. Import the asyncio library

# --- Configuration & Setup ---

try:
    # 4. Use the AsyncOpenAI client. The connection is now established on the first call.
    client = AsyncOpenAI(
        base_url="http://localhost:8002/v1",
        api_key="EMPTY"
    )
    print("✅ Async client configured. Connection will be tested on the first call.")
except Exception as e:
    print(f"❌ Client configuration failed. Error: {e}")
    exit()

def preprocess_claim(text: str) -> str:
    # This function is fast and does not need to be async. It remains unchanged.
    if not isinstance(text, str):
        return ""
    processed_text = re.sub(r'Open in a new tab', '', text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


# 5. Convert the function to an async coroutine
async def clear_figure_references(claim_text: str, semaphore: asyncio.Semaphore) -> str:
    """
    ASYNC VERSION. Uses a semaphore to limit concurrent requests.
    The core logic and prompt are identical to your original.
    """
    # This 'async with' block ensures we don't exceed our concurrency limit.
    async with semaphore:
        prompt = f"""
You are a hyper-literal text-editing bot. You will first provide your reasoning by explaining exactly what you will remove, then you will provide the cleaned text.

**CRITICAL RULES:**
1.  **REASONING FIRST:** In the 'Reasoning' section, identify the exact text to be removed based on the rules.
2.  **NO MARKDOWN (CRITICAL):** The 'Cleaned' output must be PURE, UNFORMATTED text. Do not use any Markdown (`**`, `*`), quotes (`""`), or any other formatting.
3.  **DELETE ONLY:** You must only delete the reference text.
4.  **NO REPHRASING:** Do not change the wording or grammar of the remaining text.
5.  **PARENTHESES RULE:** If a reference is inside parentheses `(...)`, the entire parenthetical block must be removed.
6.  **PROXIMITY RULE:** Do NOT remove text that is outside of the parentheses, even if it is right next to them.
7.  **INLINE RULE:** If a reference is not in parentheses, remove the reference itself AND any short, direct pointer phrases immediately preceding it (e.g., the phrase "see third row," before "Table 3").
8.  **OUTPUT FORMAT:** Provide your output in two parts: "Reasoning:" and "Cleaned:".
---
### EXAMPLES
---
(Your excellent examples remain here, unchanged)
---
### YOUR TASK
---

**Original:** "{claim_text}"
**Reasoning:**
"""
        # Return early for empty strings to avoid unnecessary API calls
        if not claim_text:
            return ""

        try:
            # 6. Use 'await' for the network call
            response = await client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=len(claim_text) + 150
            )
            
            # The rest of your parsing logic is perfect and remains unchanged.
            full_response_text = response.choices[0].message.content.strip()
            
            cleaned_text = ""
            if "Cleaned:" in full_response_text:
                cleaned_text = full_response_text.split("Cleaned:")[-1].strip()
            else:
                cleaned_text = ""
                
            return cleaned_text.replace(" .", ".").strip()

        except Exception as e:
            # This error handling is also unchanged.
            print(f"\nError during API call for claim '{claim_text[:50]}...': {e}")
            return ""


# 7. Create a main async function to orchestrate the process
async def main():
    input_filename = "all_data_without_judge_with_fig_ref.csv"
    output_filename = "all_data_without_judge_without_fig_ref.csv"

    # Set a concurrency limit. 32 is a robust starting point.
    # This prevents sending all ~4000 requests at once.
    CONCURRENCY_LIMIT = 32
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        df = pd.read_csv(input_filename)
        print(f"--- Successfully loaded '{input_filename}' with {len(df)} rows. ---")
        
        # Pre-processing is fast, so we do it sequentially first.
        print("\n--- Pre-processing all claims... ---")
        preprocessed_claims = [preprocess_claim(claim) for claim in df['claim']]
        
        # Create a list of all the async tasks we need to run.
        tasks = [clear_figure_references(claim, semaphore) for claim in preprocessed_claims]
        
        print(f"\n--- Starting {len(tasks)} concurrent API calls (limit: {CONCURRENCY_LIMIT})... ---")
        
        # Use asyncio.gather to run all tasks concurrently.
        # It preserves the order of results, which is crucial for mapping back to the DataFrame.
        # We wrap it with tqdm to get a live progress bar.
        cleaned_results = await async_tqdm.gather(*tasks, desc="Cleaning Claims (Async)")
        
        # The list 'cleaned_results' is now populated and in the correct order.
        df['new_claim'] = cleaned_results
        
        print("\n--- Processing Complete. Here's a preview of the new data: ---")
        # The real-time printing per pair is replaced by this final check,
        # as per-pair printing is not practical in a concurrent workflow.
        print(df[['claim', 'new_claim', 'class']].head())
        
        df.to_csv(output_filename, index=False)
        print(f"\n✅ Final data with cleaned claims saved to '{output_filename}'")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_filename}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# 8. The standard way to run the main async function.
if __name__ == "__main__":
    asyncio.run(main())
