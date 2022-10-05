# Databricks notebook source
import pandas as pd

# COMMAND ----------

longdesc_tickets_new = pd.read_csv('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_June2022.csv.gz') # Joined with long description and status history, ~7k
print(len(longdesc_tickets_new))
longdesc_tickets = pd.read_csv('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2.csv.gz') # Joined with long description and status history, ~450k
print(len(longdesc_tickets))
longdesc_tickets = pd.concat([longdesc_tickets, longdesc_tickets_new])
print(len(longdesc_tickets))
longdesc_tickets.drop_duplicates(subset=['TICKET_ID'], inplace=True)
print(len(longdesc_tickets))

# COMMAND ----------

all_tickets = pd.read_csv('/dbfs/FileStore/ecd/ecd_all_tickets_no_descs.csv.gz') # No joins, ~2m
all_tickets_new = pd.read_csv('/dbfs/FileStore/ecd/ecd_all_tickets_no_descs_June2022.csv.gz') # No joins, ~13k
print(len(all_tickets))
print(len(all_tickets_new))
all_tickets = pd.concat([all_tickets, all_tickets_new])
print(len(all_tickets))
all_tickets.drop_duplicates(subset=['TICKET_ID'], inplace=True)
print(len(all_tickets))

# COMMAND ----------

# Pretty sure these will not have changed in the last few weeks as they are reference tables (likely updated once per fiscal, in April)
ci_interested = pd.read_csv('/dbfs/FileStore/ecd/ASSET_ECD_CI_INTERESTED_PARTY.csv.zip', encoding='latin1') # Joinable to all_tickets on CI_NMBR
ci_info = pd.read_csv('/dbfs/FileStore/ecd/ASSET_ECD_CONFIG_ITEM.csv.zip', encoding='latin1') # Joinable to all_tickets on CI_NMBR

# COMMAND ----------

# This is the table we initially provided (lots of cleaning on the text descriptions)
print(len(longdesc_tickets), longdesc_tickets.columns)

# This is a new table with more tickets, but no text descriptions of the tickets (no cleaning)
print(len(all_tickets), all_tickets.columns)

# Add CI numbers to longdesc_tickets
longdesc_tickets = longdesc_tickets.merge(all_tickets, on='TICKET_ID')
print(len(longdesc_tickets), longdesc_tickets.columns)

# COMMAND ----------

# Drop duplicate columns
longdesc_tickets = longdesc_tickets.drop([c for c in longdesc_tickets.columns if '_y' in c or '.1' in c], axis=1)
# Rename (formerly) duplicate columns
longdesc_tickets.columns = [c[:-2] if '_x' in c else c for c in longdesc_tickets.columns]
print(len(longdesc_tickets), longdesc_tickets.columns)

# COMMAND ----------

# Config Item text descriptions, etc. Can be joined to tickets if desired
print(len(ci_info), ci_info.columns)

# Using a left join since some tickets may not have matching CIs in the ci_info table
longdesc_tickets = longdesc_tickets.merge(ci_info[['CI_NMBR', 'CI_DESC']], left_on='CONFIGURATION_ITEM_NMBR', right_on='CI_NMBR', how='left').drop(['CI_NMBR'], axis=1)
print(len(longdesc_tickets), longdesc_tickets.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC See https://adb-3053423791683952.12.azuredatabricks.net/?o=3053423791683952#notebook/4079369946232287/command/4079369946232297

# COMMAND ----------

# Interested parties for the CIs. 
print(len(ci_interested), ci_interested.columns)

# "Level 2 support" is what is used to assign tickets to Owner groups based on CI
ci_level2_support = ci_interested.query('INT_TYPE == "Level 2 Support"')[['CI_NMBR', 'PERSON_GROUP']]

# Level 2 support contains multiple entries for some CIs. We don't know which is "correct" so we'll just naively drop duplicates
print(len(ci_level2_support), 'CI->Owner group mappings, including multiple Level 2 support groups for some CIs')
ci_level2_support = ci_level2_support.drop_duplicates(subset=['CI_NMBR'])
print(len(ci_level2_support), 'CI->Owner group mappings (max 1 per CI)')

# Left join again
longdesc_tickets = longdesc_tickets.merge(ci_level2_support, left_on='CONFIGURATION_ITEM_NMBR', right_on='CI_NMBR', how='left').drop(['CI_NMBR'], axis=1)
print(len(longdesc_tickets), longdesc_tickets.columns)

# COMMAND ----------

# Rename the level 2 support column to make it more clear
longdesc_tickets.columns = ['CI_LEVEL_2_SUPPORT' if c == 'PERSON_GROUP' else c for c in longdesc_tickets.columns]

longdesc_tickets.head()

# COMMAND ----------

# Export the joined table to CSV
longdesc_tickets.to_csv('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more_withJune2022.csv', index=None, header=True)

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/ecd/ecd_tickets_cleaned_2_more_withJune2022.csv')
print(len(df))
print(df['REPORT_DATE'].min(), df['REPORT_DATE'].max())

# COMMAND ----------

df.sort_values(['REPORT_DATE']).tail(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Misc EDA during call with Curtis

# COMMAND ----------

display(longdesc_tickets.ASSIGNED_OWNER_GROUP.value_counts().to_frame().reset_index())

# COMMAND ----------

display(all_tickets.ASSIGNED_OWNER_GROUP.value_counts().to_frame().reset_index())

# COMMAND ----------

display(longdesc_tickets.query('ASSIGNED_OWNER_GROUP == "ITS00380"').astype('str'))

# COMMAND ----------

display(all_tickets.query('ASSIGNED_OWNER_GROUP == "ITS00380"').astype('str'))

# COMMAND ----------


