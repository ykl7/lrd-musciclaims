import pandas as pd
import re

def _expand_panel_range(start: str, end: str) -> str:
    if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
        start_char_code, end_char_code = ord(start.lower()), ord(end.lower())
        
        if start_char_code > end_char_code:
            return None
        
        panels = [chr(c) for c in range(start_char_code, end_char_code + 1)]
        return ', '.join(panels)
            
    return None

def extract_panel_data(text: str) -> str:
    if not isinstance(text, str):
        return None

    processed_text = text.replace('Â', ' ').replace('\xa0', ' ').replace('–', '-')

    fig_word_str = r'(?:Fig|Figure)s?\.?'
    fig_num_str = r'(?:S\d+|\d+)'

    if len(re.findall(f'{fig_word_str}\\s*({fig_num_str})', processed_text, re.IGNORECASE)) > 1:
        return 'ALL_PANELS'

    range_match = re.search(f'{fig_word_str}\\s*{fig_num_str}\\s*([a-zA-Z])\\s*-\\s*([a-zA-Z])', processed_text, re.IGNORECASE)
    if range_match:
        start_panel, end_panel = range_match.groups()
        expanded = _expand_panel_range(start_panel, end_panel)
        return expanded if expanded else None

    and_match = re.search(f'{fig_word_str}\\s*{fig_num_str}\\s*([a-zA-Z])\\s+and\\s+([a-zA-Z])\\b', processed_text, re.IGNORECASE)
    if and_match:
        panel1, panel2 = and_match.groups()
        return f"{panel1}, {panel2}".lower()

    list_match = re.search(f'{fig_word_str}\\s*{fig_num_str}\\s*([a-zA-Z](?:,\\s*(?:{fig_num_str})?[a-zA-Z])+)', processed_text, re.IGNORECASE)
    if list_match:
        panels_str = list_match.group(1)
        panels_cleaned = re.sub(r'\d+', '', panels_str)
        panels_list = [p.strip() for p in panels_cleaned.split(',')]
        return ', '.join(panels_list).lower()

    single_panel_match = re.search(f'{fig_word_str}\\s*{fig_num_str}([A-Za-z])\\b', processed_text, re.IGNORECASE)
    if single_panel_match:
        return single_panel_match.group(1).lower()

    single_panel_with_word_match = re.search(f'{fig_word_str}\\s*{fig_num_str}(?:\\s*Panel)?\\s+([A-Za-z])\\b', processed_text, re.IGNORECASE)
    if single_panel_with_word_match:
        panel = single_panel_with_word_match.group(1)
        return panel.lower()

    if re.search(f'{fig_word_str}\\s*{fig_num_str}', processed_text, re.IGNORECASE):
        return 'ALL_PANELS'

    return None

def run_debug_tests():
    print("--- Running debug tests ---")
    test_cases = {
        "(Figure 2B, 2C).": "b, c",
        "Fig. 8J-L": "j, k, l",
        "Fig. 8F and G": "f, g",
        "Fig. 5b and eqn 12": "b",
        "Fig 4A": "a",
        "Figure 1.": "ALL_PANELS",
        "Figure 1 and Fig 2.": "ALL_PANELS",
        "Figure 1A,B,C": "a, b, c",
        "Fig 1C-A": None,
    }
    all_passed = True
    for text, expected in test_cases.items():
        result = extract_panel_data(text)
        if result == expected:
            print(f"✅ PASS: '{text}' -> '{result}'")
        else:
            print(f"❌ FAIL: '{text}' -> Expected '{expected}', got '{result}'")
            all_passed = False
    print("--- Tests complete ---\n")
    return all_passed

if run_debug_tests():
    try:
        df = pd.read_csv('all_data_with_judge_with_fig_ref_v1.csv')
        df['panels'] = df['claim'].fillna('').apply(extract_panel_data)
        df.to_csv('all_data_with_judge_with_fig_ref_v2.csv', index=False)
        print(f"✅ Processed {len(df)} claims. Results saved to 'all_data_with_judge_with_fig_ref_v2.csv'")
        # print("\n--- Sample (first 20 rows) ---")
        # print(df[['claim', 'panels']].head(20).to_string())
    except FileNotFoundError:
        print("❌ Error: File not found.")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Tests failed. Fix issues before processing.")
