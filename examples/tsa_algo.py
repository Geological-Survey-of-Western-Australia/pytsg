"""
TSA/jCLST Algorithms
====================

The results of the :term:`TSA` and :term:`jCLST` unmixing algorithms commonly found in TSG datasets are split
up into a series of :term:`scalars<scalar>` which describe which minerals from the
:term:`Spectral Reference Library` contributed to the unmixing result, and the spectral contribution of each
mineral to the unknown spectrum.
"""

# %%
# Summarise jCLST results
# -----------------------
# Pivot the data so that we have a column for each mineral group which then contains the spectral
# contribution (weight) to the unmixing result.

import pandas as pd
from matplotlib import pyplot as plt

from pytsg import read_tsg

scalars = read_tsg("../example_data/GSNSW_testrocks").tir.scalars

group_columns = ["Grp1 sjCLST", "Grp2 sjCLST", "Grp3 sjCLST"]
weight_columns = ["Wt1 sjCLST", "Wt2 sjCLST", "Wt3 sjCLST"]

rows = []
for _, scalar_row in scalars.iterrows():
    row = {}
    for group, weight in zip(scalar_row[group_columns], scalar_row[weight_columns]):
        if isinstance(group, str) and group:
            row[group] = row.get(group, 0.0) + float(weight)
    rows.append(row)

jclst_summary = pd.DataFrame(rows, index=scalars.index).fillna(0.0)
# sphinx_gallery_start_ignore
pd.set_option("display.max_columns", 9)
pd.set_option("display.width", 100)
# sphinx_gallery_end_ignore
jclst_summary

# %%
# Export the summary to CSV
# -------------------------
jclst_summary.to_csv("../example_data/GSNSW_testrocks/mineral_weights.csv", index=False)

# %%
# Create summary plot
# -------------------
# Create a stacked bar plot similar to the Summary Screen in :term:`TSG`.

ax = jclst_summary.plot(kind="bar", stacked=True, width=1.0)
ax.set_xticks([])
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_title("GSNSW_testrocks sjCLST Summary")
plt.tight_layout()
plt.show()
