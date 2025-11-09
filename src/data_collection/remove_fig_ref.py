import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import re
import asyncio

client = AsyncOpenAI(
    base_url="http://localhost:8003/v1",
    api_key="EMPTY"
)
print("✅ Async client configured.")


async def validate_connection():
    try:
        await client.models.list()
        print("✅ Successfully connected to vLLM server.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False


def preprocess_claim(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'Open in a new tab', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def clear_figure_references(claim_text: str, semaphore: asyncio.Semaphore, max_retries: int = 3) -> str:
    if not claim_text:
        return ""

    # Skip figure captions
    if claim_text.lower().startswith(('figure ', 'fig ', 'table ')):
        if ':' in claim_text[:25]:
            return ""

    prompt = f"""Remove ALL figure and table references. Return ONLY cleaned text.

RULES:
1. Remove (Figure X), (Fig. X), (Table X) - entire parentheses
2. Remove "as shown in Figure X", "presented in Figure X", "in Figure X"
3. Remove "Figure X shows...", "Fig X presents..." at sentence start
4. If sentence is ONLY about a figure, return empty
5. Keep all other text unchanged

EXAMPLES:

Input: The FMEA combines brainstorming (Figure 1; Dailey, 2004).
Output: The FMEA combines brainstorming.

Input: The proportion fluctuated: 9.4% in 2010 as opposed to 25.0% in 2014 (Fig. 2).
Output: The proportion fluctuated: 9.4% in 2010 as opposed to 25.0% in 2014.

Input: From the cluster presented in Fig. 7 we selected the paper.
Output: From the cluster we selected the paper.

Input: As shown in Figure 1, we identified 935 records.
Output: We identified 935 records.

Input: As shown in Figure 2A, there were 2 retrospective database studies.
Output: There were 2 retrospective database studies.

Input: General characteristics As can be seen in Fig 2, the majority were from Turkey.
Output: General characteristics The majority were from Turkey.

Input: The green histograms in Figure 2 would be much flatter.
Output: The green histograms would be much flatter.

Input: Bond matrix is shown in Fig. 7
Output: Bond matrix is shown.

Input: Information of keywords co-occurrence. Figure 6.
Output: Information of keywords co-occurrence.

Input: Fig 1 presents a PRISMA flow diagram.
Output:

Input: Figure 1: Competing risks model of disengagement.
Output:

Input: Figure 2 shows the decline in predictive value.
Output:

Input: Shown in Figure 1 are: the tokenizer and parser.
Output: The tokenizer and parser.

Input: The database presented in Figure 2B experiences loss.
Output: The database experiences loss.

Input: Results are shown in Figure 6.
Output: Results are shown.

Input: The findings from (Figure 1) support this.
Output: The findings from support this.

Input: Supplementary information Additional file 1: Figure 1.
Output: Supplementary information Additional file 1.

Input: As illustrated by figure 1, the distribution has changed.
Output: The distribution has changed.

Input: The process is presented in Figure 1.
Output: The process is presented.

Input: Figure 6 shows the system geometry for imaging experiments.
Output:

Input: In brief, charts in Figure 1 show how in 2023 the percentage was 52%.
Output: In brief, charts show how in 2023 the percentage was 52%.

NOW CLEAN THIS (return only cleaned text, no explanations):
{claim_text}

CLEANED TEXT:"""

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="Qwen/Qwen3-VL-8B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=len(claim_text) + 100
                )
                
                result = response.choices[0].message.content.strip()
                
                # Try to extract if LLM added label
                if "CLEANED TEXT:" in result:
                    result = result.split("CLEANED TEXT:")[-1].strip()
                if "Output:" in result:
                    result = result.split("Output:")[-1].strip()
                
                # Clean artifacts
                result = result.replace(" .", ".").replace(" ,", ",").strip()
                while result and result[0] in (',', '.', ';'):
                    result = result[1:].strip()
                
                # Check if still has figure refs
                if result and re.search(r'\b(Figure|Fig|Table)\s*\d+', result, re.IGNORECASE):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        return result  # Return partial result
                
                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return 'INVALID'
        
        return 'INVALID'

def remove_figure_references(text: str) -> str:
    """Remove all figure/table references using simple regex."""
    if not isinstance(text, str):
        return text
    
    # Main pattern: Fig. X, Fig X, Figure X, Table X
    # Matches: Fig 1, Fig. 2A, Figure 3, figure 5B, Table 1, etc.
    text = re.sub(
        r'\b(?:Fig\.?|Figure|Table)s?\s*\d+[A-Za-z]?',
        '',
        text,
        flags=re.IGNORECASE
    )
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

async def main(CONCURRENCY_LIMIT: int = 32):
    if not await validate_connection():
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    try:
        df = pd.read_csv("all_data_with_judge_with_fig_ref_v2.csv")
        print(f"--- Loaded {len(df)} rows ---")
        
        print("\n--- Preprocessing... ---")
        preprocessed = [preprocess_claim(c) for c in df['claim']]
        
        print(f"\n--- Processing {len(preprocessed)} claims (limit: {CONCURRENCY_LIMIT})... ---")
        
        tasks = [clear_figure_references(c, semaphore) for c in preprocessed]
        cleaned = await async_tqdm.gather(*tasks, desc="Cleaning")
        
        df['new_claim'] = cleaned
        
        # filling nans in panels
        df.fillna({'panels': 'ALL_PANELS'}, inplace=True)

        df.dropna(subset=["new_claim"], inplace=True)
        df['new_claim'] = df['new_claim'].apply(remove_figure_references)
        
        df_valid = df[df["new_claim"] != 'INVALID'].copy()
        invalid = len(df) - len(df_valid)
        successful = sum(1 for o, n in zip(df_valid['claim'], df_valid['new_claim']) if n != o)
        
        print("\n" + "="*70)
        print(f"Total:      {len(df):,}")
        print(f"Invalid:    {invalid:,} ({invalid/len(df)*100:.1f}%)")
        print(f"Valid:      {len(df_valid):,}")
        print(f"  Cleaned:  {successful:,} ({successful/len(df_valid)*100:.1f}%)")
        print("="*70)

        count = 0
        for i in df_valid['new_claim'].tolist():
            if 'Fig' in i or 'fig' in i:
                count += 1
        print("count: ", count)
        print("Percent:", (count/len(df_valid))*100)
        
        df_valid.to_csv("all_data_with_judge_without_fig_ref_v3.csv", index=False)
        print(f"\n✅ Saved {len(df_valid):,} rows")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main(CONCURRENCY_LIMIT=250))
