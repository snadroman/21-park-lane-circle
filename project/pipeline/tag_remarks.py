"""Phrase-dictionary tagger for listing remarks.

Usage:  python pipeline/tag_remarks.py listings.csv remarks_tags.csv

Output: one row per listing keyed by ml_num, with
  a_*           one 0/1 column per condition phrase ("custom kitchen with quartz..."): 66 phrases tied to
                a condition tier, plus three untiered signals (as-is, good bones, renovator pitch)
  mot_*         seller-motivation patterns
  cond_n_atoms  number of tiered condition phrases found
"""
import re
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Atoms: name -> (tier, components, regex)
#   tier: 5 renovated/luxury, 4 updated, 3 original-but-good, 2 dated, 1 needs work,
#         None = condition signal without a tier
#   components: which component(s) the atom tells us about
# Regexes run on lowercased text with "s/s" -> "stainless steel".
# ---------------------------------------------------------------------------
ATOMS = {
    # ---- tier 5: renovated / luxury -------------------------------------------------
    "custom_kitchen_quartz": (5, ["kitchen"], r"custom kitchen|quartz counter"),
    "spa_bath_glass_shower": (5, ["bathroom"], r"spa-like bath|frameless glass shower"),
    "heated_bath_floors": (5, ["bathroom"], r"heated (bathroom |bath )?floor"),
    "designer_light_fixtures": (5, ["lighting"], r"designer light fixture"),
    "pot_lights": (5, ["lighting"], r"pot ?lights"),
    "smooth_ceilings": (5, ["finishes"], r"smooth ceilings"),
    "custom_closet_organizers": (5, ["finishes"], r"custom closet organi[sz]er"),
    "custom_window_coverings": (5, ["windows"], r"custom window covering"),
    "built_in_appliances": (5, ["appliances"], r"built-in appliances"),
    "wide_plank_hardwood": (5, ["flooring"], r"wide-plank hardwood"),
    "renovated_basement_rec": (5, ["basement"], r"(renovated|redone) basement|basement (has been |is )?(renovated|redone|finished with a rec)|basement (has|adds|includes) a rec room and an extra bath"),
    "fully_redone": (5, ["overall"], r"fully (redone|renovated)|top to bottom|totally renovated|completely (redone|renovated)|renovation is complete"),
    "turnkey": (5, ["overall"], r"turn-?key"),
    "new_roof_he_furnace": (5, ["roof", "hvac"], r"new roof and (a )?high-efficiency furnace"),
    "updated_elec_plumb": (5, ["electrical"], r"(updated|new) electrical and plumbing|electrical and plumbing (have been|are|were|has been) updated"),
    "new_windows_doors": (5, ["windows"], r"new windows and doors"),
    # ---- tier 4: updated --------------------------------------------------------------
    "kitchen_redone": (4, ["kitchen"], r"kitchen (has been |was |is )?redone|redone kitchen|kitchen redone|recently updated kitchen|updated kitchen"),
    "ss_appliances": (4, ["appliances"], r"stainless steel appliances"),
    "upgraded_kitchen_stone": (4, ["kitchen"], r"upgraded kitchen|stone counter"),
    "new_countertops": (4, ["kitchen"], r"new countertops"),
    "refreshed_bathroom": (4, ["bathroom"], r"bathroom (has been |is |was )?refreshed|refreshed bathroom|bathroom refresh"),
    "renovated_bath_vanity": (4, ["bathroom"], r"renovated bathroom|modern vanity"),
    "new_vanity": (4, ["bathroom"], r"new (bathroom )?vanity|vanity is new"),
    "fresh_paint": (4, ["finishes"], r"fresh paint|freshly painted"),
    "newer_laminate": (4, ["flooring"], r"newer laminate|new laminate"),
    "vinyl_plank": (4, ["flooring"], r"vinyl plank"),
    "refinished_hardwood": (4, ["flooring"], r"refinished hardwood|hardwood floors (are|were|have been) refinished|hardwood (is|was) refinished"),
    "newer_flooring": (4, ["flooring"], r"newer flooring|(?<!paint and )new flooring(?! throughout\b.*need)"),
    "newer_windows": (4, ["windows"], r"newer windows"),
    "newer_furnace_ac": (4, ["hvac"], r"newer furnace and central air|newer furnace"),
    "new_central_air": (4, ["hvac"], r"new central air"),
    "modern_light_fixtures": (4, ["lighting"], r"modern light fixture"),
    "rebuilt_porch": (4, ["exterior"], r"rebuilt (front )?porch"),
    "kitchen_baths_updated": (4, ["kitchen", "bathroom"], r"kitchen and baths have been updated|kitchen and bath(room)?s (were|are) updated"),
    "recent_updates": (4, ["overall"], r"recent updates|updated finishes|many modern upgrades|fully updated|thoughtfully updated|tastefully updated|updated throughout|(has|have) been updated|recently updated|nicely updated"),
    "finished_basement": (4, ["basement"], r"(?<!un)finished basement|basement is finished(?! with a rec)"),
    # ---- tier 3: original but good --------------------------------------------------
    "original_kitchen_good": (3, ["kitchen"], r"original kitchen"),
    "original_finishes_good": (3, ["finishes"], r"original finishes (in|that|are|remain)|finishes are original but|classic original finishes|mostly original finishes|original but (clean|well)"),
    "clean_cared_for": (3, ["overall"], r"clean,? (and )?cared[- ]for|well[- ]cared[- ]for|well-maintained|well maintained|well-tended|well-kept|well kept|clean and (move-in|liv(e)?able|ready|tidy)|is clean\b|it's clean\b|this clean\b|a clean\b"),
    "clean_neutral_finishes": (3, ["finishes"], r"neutral finishes|clean and neutral|finishes are clean"),
    "parquet": (3, ["flooring"], r"parquet"),
    "broadloom": (3, ["flooring"], r"broadloom"),
    "hardwood_under_broadloom": (3, ["flooring"], r"hardwood hides under|under the broadloom"),
    "original_light_fixtures": (3, ["lighting"], r"original light fixture"),
    "mechanicals_serviced": (3, ["hvac"], r"mechanicals"),
    "solid_roof": (3, ["roof"], r"solid roof|roof is solid"),
    "basement_dry_clean": (3, ["basement"], r"basement is dry|dry,? clean basement|dry and clean basement"),
    "bathrooms_older_functional": (3, ["bathroom"], r"bath(room)?s? (is|are) older but|older but functional bath|older bath(room)?s? (is|are) (clean|functional|liv)"),
    "older_bathrooms": (2, ["bathroom"], r"older bath(room)?s?\b(?! (is|are) (clean|functional|liv))|bath(room)?s? (is|are) (older|original|dated)(?! but)|original bath(room)?s?\b"),
    "furnace_older_working": (3, ["hvac"], r"furnace is older but|older but working furnace|older furnace"),
    "sound_liveable": (3, ["overall"], r"sound and liv(e)?able|liv(e)?able as-is|solid home|solid,? liv(e)?able"),
    "good_bones": (None, [], r"good bones|great bones|bones are solid|honest bones|solid bones"),
    # ---- tier 2: dated ------------------------------------------------------------------
    "original_finishes_tired": (2, ["finishes"], r"original and tired|earlier era|shows its age|dated finishes|tired finishes"),
    "older_appliances": (2, ["appliances"], r"older appliances|appliances are older|appliances (may|will|might) need replac|dated appliances"),
    "older_windows": (2, ["windows"], r"older windows|windows are older|original windows|windows (are|is) dated"),
    "could_use_updating": (2, ["overall"], r"could use (some )?updat|benefit from updates|needs? (some )?updat|some updates (are )?needed|update over time|modernize over time|in need of (some )?updat|ready for updating|updating over time|some tlc|needs? (some )?work\b|needs? some love"),
    "unfinished_basement": (2, ["basement"], r"unfinished basement"),
    # ---- tier 1: needs work -------------------------------------------------------------
    "kitchen_baths_replace": (1, ["kitchen", "bathroom"], r"kitchen and bath(room)?s (need|require)"),
    "bath_full_redo": (1, ["bathroom"], r"bathroom needs a full redo|bathroom (needs|requires) (a )?full|bathrooms need|needs a full bath|bathroom (is )?ready for a full redo"),
    "worn_flooring": (1, ["flooring"], r"worn flooring|flooring is worn"),
    "dated_electrical": (1, ["electrical"], r"electrical is dated|\bdated electrical"),
    "roof_windows_eol": (1, ["roof", "windows"], r"end of (their|its) (useful )?life"),
    "basement_waterproofing": (1, ["basement"], r"waterproofing"),
    "needs_paint_flooring": (1, ["finishes", "flooring"], r"(?<!fresh )(?<!new )paint and new flooring(?! (run|throughout make|make|give|keep))|needs? (new )?paint"),
    "full_cosmetic_overhaul": (1, ["overall"], r"cosmetic overhaul"),
    "needs_full_reno": (1, ["overall"], r"needs (a )?(full|complete|significant|major|total) (renovation|redo|update)|needs significant|full renovation|gut (job|renovation)|in need of (a )?(full|complete|major)|needs? (everything|a lot of work)|in need of replacement"),
    "bring_contractor": (1, ["overall"], r"bring your contractor"),
    "as_is": (None, [], r"as-is|as is condition|offered as is"),
    "renovator_pitch": (None, [], r"renovator|blank canvas|bring your vision|your paintbrush|sweat equity|diamond in the rough|fixer|handy"),
    # ---- neutral features ---------------------------------------------------------------
}

MOTIVATION = {
    "mot_any": r"motivated",
    "mot_offers": r"motivated.{0,40}offers?|bring your offers|invites (all |your )?offers?",
    "mot_quick_close": r"quick closing",
    "mot_flexible_close": r"flexible closing",
}

def tag(remarks: pd.Series) -> pd.DataFrame:
    """One 0/1 column per atom (a_*) and motivation pattern (mot_*), plus cond_n_atoms = number of
    tiered condition phrases found."""
    t = remarks.fillna("").str.lower().str.replace("s/s", "stainless steel", regex=False)
    out = {f"a_{name}": t.str.contains(rx, regex=True).astype(np.int8) for name, (_, _, rx) in ATOMS.items()}
    out.update({name: t.str.contains(rx, regex=True).astype(np.int8) for name, rx in MOTIVATION.items()})
    df = pd.DataFrame(out, index=remarks.index)
    df["cond_n_atoms"] = df[[f"a_{n}" for n, (tier, _, _) in ATOMS.items() if tier]].sum(1)
    return df


if __name__ == "__main__":
    warnings.simplefilter("ignore")        # pandas notes that some patterns contain groups; harmless here
    src, dst = sys.argv[1], sys.argv[2]
    d = pd.read_csv(src, usecols=["ml_num", "remarks"])
    res = tag(d["remarks"])
    res.insert(0, "ml_num", d["ml_num"].values)
    res.to_csv(dst, index=False)
    print(f"wrote {dst}: {res.shape[0]} rows x {res.shape[1]} cols")
