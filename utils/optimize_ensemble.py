import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print(" OPTIMIZING ENSEMBLE WEIGHTS VIA OOF PREDICTIONS ")
    print("=" * 60)

    # ── Paths (Modify if your local paths are different) ─────────────
    # Assuming Kaggle output dir for OOF, but you may need to point this to your local path
    oof_path = Path("output/submission_outputs/cv_oof_predictions.parquet")
    if not oof_path.exists():
        # Fallback to local E: drive structure just in case
        oof_path = Path("cv_oof_predictions.parquet") 
        
    sku_feat_path = Path("output/feature_outputs/sku_features.csv")

    if not oof_path.exists():
        print(f"❌ Error: Could not find OOF file at {oof_path}")
        print("Please run step2_train.py with run_cv_flag=True first!")
        return
    if not sku_feat_path.exists():
        print(f"❌ Error: Could not find SKU features at {sku_feat_path}")
        return

    print("1. Loading OOF predictions and SKU features...")
    oof = pd.read_parquet(oof_path)
    sf = pd.read_csv(sku_feat_path)

    print(f"   -> Loaded {len(oof):,} OOF prediction rows.")

    # 2. Merge taxonomy
    df = oof.merge(
        sf[["ItemCode", "profit_tier_enc", "demand_density_enc", "ets_available", "profit_weight"]],
        on="ItemCode", how="left"
    )

    # 3. Define the objective function (Minimize Weighted Mean Squared Error)
    def objective(weights, true_y, pred_A, pred_D, pred_E, sku_weights):
        wA, wD, wE = weights
        blend = wA * pred_A + wD * pred_D + wE * pred_E
        sq_error = (true_y - blend) ** 2
        return np.sum(sq_error * sku_weights)
    
    # Constraints: weights must sum to 1.0
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds_all = [(0, 1), (0, 1), (0, 1)] # A, D, E
    bounds_no_ets = [(0, 1), (0, 1), (0, 0)] # Force E to 0%

    # 4. Calculate horizon day and split into w14 (Days 1-28) and w58 (Days 29-56)
    df["horizon_day"] = df.groupby(["fold_id", "ItemCode"]).cumcount() + 1
    
    print("\n2. Optimizing Blends per Stratum (Split by w14 vs w58):")
    print("-" * 75)
    
    tier_names = {0: "Tier A", 1: "Tier B", 2: "Tier C", 3: "Tier D"}
    density_names = {0: "Dense", 1: "Intermittent", 2: "Sparse"}

    for tier in [0, 1, 2, 3]:
        for density in [0, 1, 2]:
            mask = (df["profit_tier_enc"] == tier) & (df["demand_density_enc"] == density)
            sub_full = df[mask]
            
            if len(sub_full) == 0: 
                continue
            
            t_name = tier_names.get(tier, f"T{tier}")
            d_name = density_names.get(density, f"D{density}")
            print(f"\n{t_name} | {d_name}:")
            
            for split_name, split_mask in [("w14 (Weeks 1-4)", sub_full["horizon_day"] <= 28),
                                           ("w58 (Weeks 5-8)", sub_full["horizon_day"] > 28)]:
                sub = sub_full[split_mask]
                if len(sub) == 0:
                    continue
                    
                # Initial guess: 100% LightGBM
                init_w = [1.0, 0.0, 0.0]
                
                # ETS is generally only allowed/useful for Dense SKUs
                bnds = bounds_all if density == 0 else bounds_no_ets
                    
                res = minimize(
                    objective, 
                    init_w, 
                    args=(
                        sub["actual"].values, 
                        sub["pred_A"].values, 
                        sub["pred_D"].values, 
                        sub["pred_E"].values, 
                        sub["profit_weight"].values
                    ),
                    method='SLSQP', 
                    bounds=bnds, 
                    constraints=cons
                )
                
                wA, wD, wE = res.x
                
                print(f"  {split_name:<15} => "
                      f"A: {wA*100:>4.1f}% | D: {wD*100:>4.1f}% | E: {wE*100:>4.1f}%")

    print("-" * 75)
    print("Done! You can use these percentages to update TIER_BLEND in step2_train.py.")

if __name__ == "__main__":
    main()
