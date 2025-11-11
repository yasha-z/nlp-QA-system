import pandas as pd
import os

src = r"E:\NLP\asap-sas\train.tsv"
dst = r"E:\NLP\data\train.csv"
print('Reading', src)
df = pd.read_csv(src, sep='\t')
print('Columns:', df.columns.tolist())

# Choose score: prefer Score1 if present, else Score2, else use any Score* column
if 'Score1' in df.columns:
    df['score'] = df['Score1']
elif 'Score2' in df.columns:
    df['score'] = df['Score2']
else:
    score_cols = [c for c in df.columns if 'Score' in c]
    if score_cols:
        df['score'] = df[score_cols[0]]
    else:
        raise SystemExit('No score column found in source file')

# question_text: use EssaySet if present
if 'EssaySet' in df.columns:
    df['question_text'] = df['EssaySet'].astype(str)
else:
    df['question_text'] = ''

# student answer text
if 'EssayText' in df.columns:
    df['student_answer'] = df['EssayText']
elif 'Answer' in df.columns:
    df['student_answer'] = df['Answer']
else:
    raise SystemExit('No student answer column found in source file')

# model_answer: leave blank for now
df['model_answer'] = ''

# write normalized CSV
out = df[['question_text','model_answer','student_answer','score']]
os.makedirs(os.path.dirname(dst), exist_ok=True)
out.to_csv(dst, index=False)
print('Wrote', dst, 'with', len(out), 'rows')
print(out.head().to_string(index=False))
