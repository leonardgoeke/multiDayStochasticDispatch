import pypsa
import pandas as pd

# -------------------------------------------------
# countries to keep
# -------------------------------------------------

countries = ["FR", "DE", "CH"]


# -------------------------------------------------
# load network
# -------------------------------------------------

n = pypsa.Network("preprocessing/data/gridData/elec_s_1024_ec.nc")

# -------------------------------------------------
# keep only AC buses in selected countries
# -------------------------------------------------

valid_buses = n.buses[(n.buses.carrier == "AC") & (n.buses.country.isin(countries))].index

# -------------------------------------------------
# create reduced network
# -------------------------------------------------

n_sub = n.copy()

# buses
n_sub.buses = n.buses.loc[valid_buses].copy()

# -------------------------------------------------
# keep only lines connecting remaining buses
# -------------------------------------------------

n_sub.lines = n.lines[n.lines.bus0.isin(valid_buses) & n.lines.bus1.isin(valid_buses)].copy()

# -------------------------------------------------
# keep only links connecting remaining buses
# -------------------------------------------------

n_sub.links = n.links[n.links.bus0.isin(valid_buses) & n.links.bus1.isin(valid_buses)].copy()

# -------------------------------------------------
# keep only transformers connecting remaining buses
# -------------------------------------------------

n_sub.transformers = n.transformers[n.transformers.bus0.isin(valid_buses) & n.transformers.bus1.isin(valid_buses)].copy()

# -------------------------------------------------
# IMPORTANT:
# remove orphaned components
# -------------------------------------------------

for comp in ["generators", "loads", "stores", "storage_units"]:
    df = getattr(n_sub, comp)

    if "bus" in df.columns:
        setattr(
            n_sub,
            comp,
            df[df.bus.isin(valid_buses)].copy()
        )

# -------------------------------------------------
# sanity checks
# -------------------------------------------------

print("Countries:")
print(n_sub.buses.country.unique())

print("\nCarriers:")
print(n_sub.buses.carrier.unique())

print("\nNumber of buses:")
print(len(n_sub.buses))

# -------------------------------------------------
# cluster by country
# -------------------------------------------------

busmap = n_sub.buses.country.to_dict()

nc = n_sub.cluster.cluster_by_busmap(busmap)

# -------------------------------------------------
# aggregated buses
# -------------------------------------------------

print("\nAggregated buses:")
print(nc.buses.index)

# -------------------------------------------------
# PTDF
# -------------------------------------------------

subnet = nc.sub_networks.obj[0]

subnet.calculate_PTDF()

# bus0/bus1 for each branch, in the same row order as subnet.branches_i() /
# subnet.PTDF, so PTDF rows can be unambiguously tied back to a country pair
branch_df = subnet.branches().loc[subnet.branches_i(), ["bus0", "bus1"]].reset_index(drop=True)

ptdf = pd.DataFrame(
    subnet.PTDF,
    columns=subnet.buses_i()
).reset_index(drop=True)

# -------------------------------------------------
# capacity per branch (same row order as ptdf/branch_df)
# -------------------------------------------------

branch_df["capacity_mw"] = (
    subnet.branches()
    .loc[subnet.branches_i(), "s_nom"]
    .reset_index(drop=True)
)

# -------------------------------------------------
# merge branch identity + capacity + PTDF coefficients
# -------------------------------------------------

grid = pd.concat([branch_df, ptdf], axis=1)

print("\nMerged branch/capacity/PTDF table:")
print(grid)

# -------------------------------------------------
# save output
# -------------------------------------------------

grid.to_csv("modelData/basis/par_ptdf.csv", index=False)

print("\nSaved:")
print(" - modelData/basis/par_ptdf.csv")