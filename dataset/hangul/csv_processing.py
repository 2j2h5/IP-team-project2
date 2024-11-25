import pandas as pd

# Define file path and load the data
syllables = [
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머',
    '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
    '소', '오', '조', '구', '누', '두', '루', '무', '부', '수',
    '우', '주', '아', '바', '사', '자', '배', '하', '허', '호',
    '국', '합', '육', '해', '공', '외', '교', '영', '준', '기',
    '협', '정', '대', '표'
]

file_path = 'labels_hangul.csv'  # 수정된 경로 입력
labels_df = pd.read_csv(file_path)

# Create a mapping dictionary
syllable_to_index = {syllable: index for index, syllable in enumerate(syllables)}

# Function to map each character individually
def map_single_char(row):
    if row in syllable_to_index:
        return syllable_to_index[row]
    else:
        return -1  # Handle unexpected characters

# Apply the function to map each syllable to its index
labels_df['Mapped'] = labels_df.iloc[:, 0].map(map_single_char)

# Drop the original column for simplicity if needed
labels_df = labels_df[['Mapped']]

# Save the result (optional)
labels_df.to_csv('labels.csv', index=False)