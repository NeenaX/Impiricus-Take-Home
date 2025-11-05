import pandas as pd

CAB_FILEPATH = '../data/cab_courses.json'
BULLETIN_FILEPATH = '../data/bulletin_courses.csv'

cab = pd.read_json(CAB_FILEPATH)
bulletin = pd.read_csv(BULLETIN_FILEPATH)

# Create a department column for bulletin
bulletin['department'] = bulletin['course_code'].str.split().str[0]

# Outer merge to combine the dfs
merged = pd.merge(cab, bulletin, on=list(bulletin.columns), how='outer') 

merged.to_csv('../data/courses.csv', index=False)