import pandas as pd
import random
from tqdm.auto import tqdm

def create_neutral_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 'NEUTRAL' data points by swapping figures within the same paper.

    For each claim in the input DataFrame, this function creates a new 'NEUTRAL'
    claim by pairing it with a different figure from the same paper.

    Args:
        df: A DataFrame containing at least 'paper_id', 'figure_id', 'claim',
            'title', 'caption', 'local_image_path', 'url', and 'class'.

    Returns:
        A new DataFrame containing the original data plus the new 'NEUTRAL' rows.
    """
    print("--- Starting Neutral Pair Generation ---")

    # 1. Identify papers with more than one unique figure. These are our candidates for swapping.
    
    figure_counts_per_paper = df.groupby('paper_id')['figure_id'].nunique()
    eligible_papers = figure_counts_per_paper[figure_counts_per_paper > 1].index.tolist()
    
    if not eligible_papers:
        print("❌ No papers found with more than one figure. Cannot generate neutral pairs.")
        return df

    print(f"✅ Found {len(eligible_papers)} papers with multiple figures eligible for swapping.")

    # 2. Filter the DataFrame to only include rows from these eligible papers.
    df_eligible = df[df['paper_id'].isin(eligible_papers)].copy()
    print(f"Working with {len(df_eligible)} claims from eligible papers.")

    # 3. Create efficient lookups to speed up the process.
    # Map each paper to its list of unique figures.
    paper_to_figures_map = df_eligible.groupby('paper_id')['figure_id'].unique().apply(list).to_dict()
    
    # Map each unique figure to its metadata (title, caption, etc.).
    figure_info_map = df_eligible.drop_duplicates(subset='figure_id').set_index('figure_id')
    figure_metadata_cols = ['title', 'caption', 'local_image_path', 'url']
    figure_info_map = figure_info_map[figure_metadata_cols].to_dict('index')

    new_neutral_rows = []
    print("\nGenerating neutral pairs by swapping figures...")

    # 4. Iterate through each eligible claim to create a neutral counterpart.
    for index, row in tqdm(df_eligible.iterrows(), total=df_eligible.shape[0]):
        current_paper_id = row['paper_id']
        current_figure_id = row['figure_id']
        
        # Find all possible figures for this paper, excluding the current one.
        swap_options = [fig_id for fig_id in paper_to_figures_map[current_paper_id] if fig_id != current_figure_id]
        
        # If there are valid figures to swap with...
        if swap_options:
            # Randomly select a new target figure.
            target_figure_id = random.choice(swap_options)
            
            # Create a new row based on the original claim.
            new_row = row.to_dict()
            
            # Get the metadata for the target figure.
            target_figure_metadata = figure_info_map[target_figure_id]
            
            # --- This is the SWAP logic ---
            # Overwrite the figure information with the target figure's info.
            new_row['figure_id'] = target_figure_id
            new_row['title'] = target_figure_metadata['title']
            new_row['caption'] = target_figure_metadata['caption']
            new_row['local_image_path'] = target_figure_metadata['local_image_path']
            new_row['url'] = target_figure_metadata['url']
            # -----------------------------
            
            # Set the class for this new, mismatched pair to NEUTRAL.
            new_row['class'] = 'NEUTRAL'
            
            new_neutral_rows.append(new_row)

    if not new_neutral_rows:
        print("⚠️ No neutral rows were generated.")
        return df

    # 5. Combine the original DataFrame with the new neutral rows.
    neutral_df = pd.DataFrame(new_neutral_rows)
    # final_df = pd.concat([df, neutral_df], ignore_index=True)
    
    print(f"\n✅ Successfully generated {len(neutral_df)} new 'NEUTRAL' data points.")
    return neutral_df

# --- Main Execution ---

if __name__ == "__main__":
    # Load the dataset created in the previous step.
    input_filename = 'all_support_data.csv'
    output_filename = 'all_neutral_data.csv'
    
    try:
        df_original = pd.read_csv(input_filename)
        print(f"Loaded '{input_filename}' with {len(df_original)} rows.")
        print("\nOriginal class distribution:")
        print(df_original['class'].value_counts())
        
        # Run the logic to create neutral data points.
        df_final = create_neutral_pairs(df_original)
        
        # Display the final class distribution.
        print("\nFinal class distribution:")
        print(df_final['class'].value_counts())
        
        # Save the final, complete dataset.
        df_final.to_csv(output_filename, index=False)
        print(f"\n🎉 Success! Final dataset with {len(df_final)} rows saved to '{output_filename}'.")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_filename}'. Please run the previous script first.")

