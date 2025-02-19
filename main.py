import re
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import WordCloud
import os

# Read file and output the full text
with open(r'data.txt', mode='r', encoding='utf-8') as f:
    txt = f.read()
print("Full text content:")
print(txt)

# Use positive lookahead to split records: each record begins with a line that includes "username date time"
records = re.split(r'(?=^[^\n]+\d{4}/\d{1,2}/\d{1,2}\s\d{1,2}:\d{2}:\d{2})', txt, flags=re.M)
records = [r.strip() for r in records if r.strip() != '']
print("Record count:", len(records))

# Parse each record, using the first line as the header and the rest as content, filtering out content within 【】
data_list = []
for rec in records:
    lines = rec.splitlines()
    header = lines[0].strip()
    # Remove the 【】 at the beginning of the header along with its content
    header = re.sub(r'^【[^】]*】', '', header)
    parts = header.split()
    # If the parsed result does not match the "username date time" format, skip this record
    if len(parts) < 3:
        continue
    name = parts[0]
    date_str = parts[1]
    time_val = parts[2]
    # Concatenate the remaining lines as the chat content, filtering out the 【】 at the beginning
    content = ''.join(lines[1:]).strip()
    content = re.sub(r'^【[^】]*】', '', content)
    data_list.append({
         'name': name,
         'date': date_str,
         'time_str': time_val,
         'content': content
    })

df = pd.DataFrame(data_list)
print("Parsed DataFrame:")
print(df)

# Convert time: combine date and time into a complete time string, then convert to datetime type
df['time'] = pd.to_datetime(df['date'] + " " + df['time_str'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
# Filter out records where conversion failed
df = df.dropna(subset=['time'])
# Keep the date as a string in the DataFrame (or directly use the date part from time)
df['date'] = df['time'].dt.date

# Set QQ as the username (process further if needed)
df['QQ'] = df['name']

print("Cleaned DataFrame:")
print(df)

# The following statistics and plotting code remains unchanged
print("Chat counts per QQ number:")
print(df['QQ'].value_counts())

df['weekday'] = df['time'].dt.weekday + 1
weekday_counts = df['weekday'].value_counts().sort_index()
print("Posts count per weekday:")
print(weekday_counts)
if not weekday_counts.empty:
    weekday_counts.plot(kind='bar')
    plt.title("Posts count per weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Post count")
    # plt.show()
else:
    print("No data available to plot weekday distribution.")

# Generate a deduplicated DataFrame (deduplicated by QQ and date)
df1 = df.drop_duplicates(subset=['QQ', 'date'])[['QQ', 'date']]
df1.reset_index(drop=True, inplace=True)
df1['date'] = pd.to_datetime(df1['date'])
df1.sort_values(by=['QQ', 'date'], inplace=True)
df1.reset_index(drop=True, inplace=True)
print("Deduplicated DataFrame df1:")
print(df1)

# Set the QQ column of df1 as the index
df1_group = df1.set_index('QQ')

def calculate_consecutive_days(group):
    if len(group) > 1:
        consecutive_days = group['date'].diff().dt.days
        consecutive_flag = consecutive_days.fillna(0) != 1
        group_id = consecutive_flag.cumsum()
        max_consecutive_days = group_id.value_counts().max()
        id_max_consecutive_days = group_id.value_counts().idxmax()
        group_subset = group[group_id == id_max_consecutive_days]
        start_date = group_subset.iloc[0]['date'].strftime('%Y-%m-%d')
        end_date = group_subset.iloc[-1]['date'].strftime('%Y-%m-%d')
        return pd.Series([max_consecutive_days, start_date, end_date],
                         index=['Longest consecutive chat days', 'Start date', 'End date'])
    elif len(group) == 1:
        date = group['date'].iloc[0].strftime('%Y-%m-%d')
        return pd.Series([1, date, date], index=['Longest consecutive chat days', 'Start date', 'End date'])
    else:
        return pd.Series([0, None, None], index=['Longest consecutive chat days', 'Start date', 'End date'])

result = df1_group.groupby(level=0, group_keys=False).apply(calculate_consecutive_days).reset_index()
print("Consecutive chat days per QQ:")
print(result)

content_txt = df['content'].str.cat(sep='。')
with open(os.path.join(os.path.dirname(__file__), 'stopwords.txt'), encoding='utf-8') as file:
    stopword = file.read()
stop_list = stopword.splitlines()

content_txt = content_txt.replace('[Image]', '').replace('[Emoji]', '')
# Get all user names (as a set)
name_set = set(df['name'].unique())
# After tokenizing with jieba, filter out stop words, single-character words, words containing digits, and user names
words = jieba.lcut(content_txt)
words = [word for word in words 
         if (word not in stop_list) 
         and (len(word) > 1) 
         and (not re.search(r'\d', word))
         and (word not in name_set)]
word_s = pd.Series(words).value_counts()
print("Word frequency count:")
print(word_s.head())

data = word_s.reset_index().values.tolist()
wordcloud = (
    WordCloud()
    .add(series_name="QQ", data_pair=data, word_size_range=[20, 66])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="QQ", title_textstyle_opts=opts.TextStyleOpts(font_size=23)),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)
wordcloud.render("QQ_wordcloud.html")
print("Word cloud generated: QQ_wordcloud.html")
