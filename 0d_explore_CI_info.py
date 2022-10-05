# Databricks notebook source
# MAGIC %md
# MAGIC # Explore Config Item info
# MAGIC Examine the EDR tables relating to CIs. Do they provide any useful information for our classifier?

# COMMAND ----------

spark.conf.set("fs.azure.account.key.scsccsadsailabdevdls1.dfs.core.windows.net", dbutils.secrets.get(scope="storage-account-access-key", key="storage-account-access-key"))

import pandas as pd

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## CONFIG_ITEM
# MAGIC No status history, therefore it won't help us resolve the change in Owner Groups over time.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from edr.ASSET_ECD_CONFIG_ITEM limit 5

# COMMAND ----------

# No status change in this table. CI numbers are unique.
ci = spark.sql('select * from edr.ASSET_ECD_CONFIG_ITEM').toPandas()
ci['CI_NMBR'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## CI_ATTRIBUTES, CI_CLASSIFICATION
# MAGIC Not useful

# COMMAND ----------

# MAGIC %sql
# MAGIC --Not useful
# MAGIC select * from edr.asset_ecd_ci_attributes limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC --Not useful
# MAGIC select * from edr.asset_ecd_ci_classification limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## CI_INTERESTED_PARTY
# MAGIC This table indicates the **level 2 support groups** (the Owner Group that service desk agents assign tickets to for this CI).
# MAGIC 
# MAGIC A few of the CIs have multiple level 2 support groups. Not clear how this helps us but we can do some analysis on historical ticket assignments.

# COMMAND ----------

ci_int_party = spark.sql('select * from edr.asset_ecd_ci_interested_party').toPandas()
ci_owner_group = ci_int_party.query('INT_TYPE == "Level 2 Support"')[['CI_NMBR', 'PERSON_GROUP']]
# Mostly these are unique.. but not all
ci_owner_group['CI_NMBR'].value_counts().hist()

# COMMAND ----------

# Here's one that's not unique. What to do about this?
ci_int_party.query('CI_NMBR == "5677158" and INT_TYPE == "Level 2 Support"')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level 2 support?
# MAGIC Are tickets typically resolved by the level 2 support group suggested by the CI? **Surprisingly, no!** This might have to do with the CI level 2 support groups changing over time, incorrect CI assignment, or the status history of CI assignment not being recorded with tickets.
# MAGIC 
# MAGIC Running the following cells while changing the date range (`ticket.REPORT_DATE >= ...`) demonstrates that more recent date ranges have higher accuracy based on level 2 support group (around 40% when starting in FY16/17... around 60% when starting in FY21/22)
# MAGIC 
# MAGIC Based on this, I conclude that re-mapping resolver groups based on level 2 support for CIs is not a good way to address the changing resolver groups over time.

# COMMAND ----------

# Analysis: Does the Level 2 support group correspond to ticket's resolver group?
ticket_resolvers = spark.sql('''
    SELECT
        ticket.TICKET_ID,
        ticket.CONFIGURATION_ITEM_NMBR as CI_NMBR,
        status.ASSIGNED_OWNER_GROUP
    FROM
        edr.DEMAND_ECD_TICKET_STATUS_HSTRY AS status
    INNER JOIN
        edr.DEMAND_ECD_TICKET as ticket ON status.ticket_id = ticket.ticket_id
    WHERE
        ticket_status_history_id IN (
            SELECT
                max(ticket_status_history_id)
            FROM
                edr.DEMAND_ECD_TICKET_STATUS_HSTRY
            WHERE
                assigned_owner_group <> 'ESI00011'
                and
                assigned_owner_group <> 'ESI00043'
            GROUP BY
                ticket_id
        )
        AND ticket.STATUS in ('RESOLVED', 'CLOSED')
        AND ticket.REPORT_DATE >= '2016-04-01'
--         AND ticket.REPORT_DATE < '2022-01-01'
--         AND ticket.DEPT_GC_ORG_NAME_EN_STD = 'Treasury Board of Canada Secretariat'
''').toPandas()

# COMMAND ----------

ci_level_2_is_unique = (ci_owner_group['CI_NMBR'].value_counts() == 1).to_frame().reset_index()
ci_level_2_is_unique.columns = ['CI_NMBR', 'UNIQUE_LEVEL_2']
ticket_resolvers = ticket_resolvers.merge(ci_level_2_is_unique, how='left')

# COMMAND ----------

# We don't know which Level 2 support group is unique, but we only want 1 (doesn't matter which)
print(len(ci_owner_group))
ci_owner_group_nodupes = ci_owner_group.drop_duplicates(subset=['CI_NMBR'])
print(len(ci_owner_group_nodupes))

# COMMAND ----------

ci_owner_group_nodupes.columns = ['CI_NMBR', 'LEVEL_2_SUPPORT']
ticket_resolvers = ticket_resolvers.merge(ci_owner_group_nodupes, how='left')

# COMMAND ----------

ticket_resolvers['RESOLVER_IS_LEVEL_2'] = ticket_resolvers['ASSIGNED_OWNER_GROUP'] == ticket_resolvers['LEVEL_2_SUPPORT']
ticket_resolvers.head()

# COMMAND ----------

len(ticket_resolvers)

# COMMAND ----------

# Given that the CI Level 2 support group is supposed to be the correct resolver group..
#  I would expect more tickets to have been resolved by the level 2 support group.
# Perhaps this is due to the changing CI mappings of level 2 support over time?
# The ESD staff have indicated that CIs are not to be trusted though. We don't know if they change over time (no history of CI assignment to ticket)
print(f'{100 * ticket_resolvers["LEVEL_2_SUPPORT"].isna().sum() / len(ticket_resolvers):.2f}% tickets have no level 2 support group indicated')
print(f'{100 * (1 - ticket_resolvers["UNIQUE_LEVEL_2"].sum() / len(ticket_resolvers)):.2f}% of tickets have CI with non-unique level 2 support group')
print(f"{100 * ticket_resolvers['LEVEL_2_SUPPORT'].isin(['ESI00011', 'ESI00043']).sum() / len(ticket_resolvers):.2f}% of tickets have level 2 support group of ESI11 or ESI43")
print(f'{100 * ticket_resolvers["RESOLVER_IS_LEVEL_2"].sum() / len(ticket_resolvers):.2f}% resolved by level 2 support group')

# COMMAND ----------

# MAGIC %md
# MAGIC ## CI_RELATION, CI_SPECIFICATION, CI_STATUS_HISTORY, CLASS_STRUCTURE
# MAGIC Not useful.

# COMMAND ----------

# Not useful
ci_rel = spark.sql('select * from edr.asset_ecd_ci_relation').toPandas()
ci_rel['CI_RELATN_TYPE'].value_counts()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Not useful
# MAGIC select * from edr.asset_ecd_ci_specification limit 5

# COMMAND ----------

ci_hstry = spark.sql('select * from edr.asset_ecd_ci_status_history').toPandas()
# Once again, there are few CI's that have multiple history entries
ci_hstry['CI_NMBR'].value_counts().hist()

# COMMAND ----------

# Here's one that does.
# Not useful
ci_hstry.query('CI_NMBR == "1392730"')

# COMMAND ----------

# MAGIC %sql
# MAGIC --Not useful
# MAGIC select * from edr.asset_ecd_class_structure limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Something else

# COMMAND ----------


