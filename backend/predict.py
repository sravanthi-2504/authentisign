    """
    predict.py  —  AuthentiSign  |  Contrastive-Loss Siamese Model
    ==============================================================
    ✅ Works with the new contrastive loss model from train_model.py
       Model outputs EUCLIDEAN DISTANCE between embeddings:
         distance < 0.5  →  GENUINE
         distance ≥ 0.5  →  FORGED
    
    Usage (CLI):
      python predict.py --original sig1.png --test sig2.png
      python predict.py --batch pairs.csv --output results.csv
      python predict.py --evaluate ../dataset/cedar_dataset --num-pairs 200
      python predict.py --original sig1.png --test sig2.png --json
    
    Usage (module):
      from predict import verify
      r = verify("original.png", "test.png")
      print(r["status"])      # "GENUINE" or "FORGED"
      print(r["distance"])    # e.g. 0.12  (genuine pairs close to 0)
      print(r["confidence"])  # e.g. 88.0
    """

    import os, sys, random, json, csv, time, argparse
    import numpy as np
    import cv2
    from pathlib import Path
    from datetime import datetime

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    GREEN  = "\033[92m"; RED   = "\033[91m"
    YELLOW = "\033[93m"; CYAN  = "\033[96m"
    BOLD   = "\033[1m";  RESET = "\033[0m"

    # Margin used during training (must match train_model.py MARGIN)
    MARGIN             = 1.0
    DECISION_THRESHOLD = 0.8  # 0.5


    # ════════════════════════════════════════════════════════════════════════
    #  AbsoluteLayer — module level (same in train_model.py + app.py)
    # ════════════════════════════════════════════════════════════════════════
    class AbsoluteLayer(layers.Layer):
        def call(self, inputs):   return tf.abs(inputs)
        def get_config(self):     return super().get_config()


    # ════════════════════════════════════════════════════════════════════════
    #  Rebuild model architecture (for weights-only loading)
    # ════════════════════════════════════════════════════════════════════════


    # ════════════════════════════════════════════════════════════════════════
    #  Model loading
    # ════════════════════════════════════════════════════════════════════════
    def load_model(model_path=None):
        if model_path is None:
            script_dir = Path(__file__).parent
            model_path = str(script_dir / "signature_model_final.keras")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun python train_model.py first."
            )

        print(f"  Loading model → {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("  ✓ Model loaded successfully")
        return model


    # ════════════════════════════════════════════════════════════════════════
    #  Preprocessing — IDENTICAL to train_model.py & app.py
    # ════════════════════════════════════════════════════════════════════════
    def preprocess(image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        img = cv2.resize(img, (128, 128))
        img = cv2.bilateralFilter(img, 9, 75, 75)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        img    = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img    = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=-1)


    # ════════════════════════════════════════════════════════════════════════
    #  Core prediction
    # ════════════════════════════════════════════════════════════════════════
    def predict_pair(model, original_path, test_path, threshold=0.8):
        img1 = np.expand_dims(preprocess(original_path), 0)
        img2 = np.expand_dims(preprocess(test_path), 0)

        prob = float(model.predict([img1, img2], verbose=0)[0][0])

        is_genuine = prob > threshold
        confidence = prob * 100 if is_genuine else (1 - prob) * 100

        return {
            "status": "GENUINE" if is_genuine else "FORGED",
            "confidence": round(confidence, 2),
            "probability": round(prob, 4),
        }


    # ════════════════════════════════════════════════════════════════════════
    #  Pretty print
    # ════════════════════════════════════════════════════════════════════════
    def print_result(r):
        col  = GREEN if r["status"] == "GENUINE" else RED
        icon = "✓" if r["status"] == "GENUINE" else "✗"

        print("\n" + "═"*55)
        print("  SIGNATURE VERIFICATION RESULT")
        print("═"*55)
        print(f"  Status      : {col}{r['status']} {icon}{RESET}")
        print(f"  Confidence  : {r['confidence']}%")
        print(f"  Probability : {r['probability']}")
        print("═"*55 + "\n")



    # ════════════════════════════════════════════════════════════════════════
    #  Batch prediction
    # ════════════════════════════════════════════════════════════════════════
    def batch_predict(model, csv_path, output_csv=None, threshold=DECISION_THRESHOLD):
        rows = []; correct = total = 0; has_labels = False

        with open(csv_path, newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0 and row[0].strip().lower() in ("original_path","original","orig"):
                    continue
                if len(row) < 2: continue
                orig, test = row[0].strip(), row[1].strip()
                label = int(row[2].strip()) if len(row) >= 3 else None

                print(f"  [{i+1:>3}] {Path(orig).name:<30} vs {Path(test).name:<30}", end="  →  ")
                try:
                    res  = predict_pair(model, orig, test, threshold)
                    col  = GREEN if res["status"]=="GENUINE" else RED
                    print(f"{col}{res['status']}{RESET} (prob={res['probability']}, conf={res['confidence']}%)", end="")
                    if label is not None:
                        has_labels = True
                        pred = 1 if res["status"]=="GENUINE" else 0
                        ok   = pred == label
                        correct += int(ok); total += 1
                        print(f"  {GREEN+'✓' if ok else RED+'✗'}{RESET}", end="")
                    print(); res["true_label"] = label; rows.append(res)
                except Exception as e:
                    print(f"{RED}ERROR: {e}{RESET}")

        print(f"\n{'─'*55}")
        print(f"  Pairs: {len(rows)}")
        if has_labels and total:
            acc = correct/total*100
            print(f"  Accuracy: {GREEN if acc>=80 else RED}{acc:.2f}%{RESET}")
        print(f"{'─'*55}")

        if output_csv and rows:
            with open(output_csv,"w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f"\n  Saved → {output_csv}")
        return rows


    # ════════════════════════════════════════════════════════════════════════
    #  Dataset evaluation
    # ════════════════════════════════════════════════════════════════════════
    def evaluate_dataset(model, dataset_path, num_pairs=100, threshold=DECISION_THRESHOLD):
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        except ImportError:
            print(f"{RED}Install scikit-learn: pip install scikit-learn{RESET}"); return {}

        org  = Path(dataset_path)/"full_org"
        forg = Path(dataset_path)/"full_forg"
        if not org.exists(): print(f"{RED}Dataset not found{RESET}"); return {}

        writers = {}
        for p in org.glob("*.png"):
            parts = p.stem.split("_")
            if len(parts)<3: continue
            wid = parts[1]
            writers.setdefault(wid,{"genuine":[],"forged":[]})
            writers[wid]["genuine"].append(p)
        for p in forg.glob("*.png"):
            parts = p.stem.split("_")
            if len(parts)<3: continue
            wid = parts[1]
            if wid in writers: writers[wid]["forged"].append(p)

        y_true=[]; y_pred=[]; half=num_pairs//2; gc=fc=0
        print(f"\n  Evaluating {num_pairs} pairs …\n")

        for wid,data in writers.items():
            if gc>=half: break
            gen=data["genuine"]
            if len(gen)<2: continue
            for i in range(min(3,len(gen)-1)):
                if gc>=half: break
                try:
                    r=predict_pair(model,gen[i],gen[i+1],threshold)
                    y_true.append(1); y_pred.append(1 if r["status"]=="GENUINE" else 0)
                    gc+=1
                    print(f"  [G] {gen[i].name:<35} p={r['probability']:.3f} → {r['status']}")
                except: pass

        for wid,data in writers.items():
            if fc>=half: break
            gen=data["genuine"]; forg_list=data["forged"]
            if not gen or not forg_list: continue
            for f in forg_list[:3]:
                if fc>=half: break
                try:
                    r=predict_pair(model,gen[0],f,threshold)
                    y_true.append(0); y_pred.append(1 if r["status"]=="GENUINE" else 0)
                    fc+=1
                    print(f"  [F] {f.name:<35} d={r['distance']:.3f} → {r['status']}")
                except: pass

        if not y_true: print(f"{RED}No pairs evaluated.{RESET}"); return {}

        acc  = accuracy_score(y_true,y_pred)*100
        prec = precision_score(y_true,y_pred,zero_division=0)*100
        rec  = recall_score(y_true,y_pred,zero_division=0)*100
        f1   = f1_score(y_true,y_pred,zero_division=0)*100
        cm   = confusion_matrix(y_true,y_pred)

        def c(v): return GREEN if v>=80 else RED
        print(f"\n{'═'*55}")
        print(f"  {BOLD}EVALUATION RESULTS{RESET}  ({len(y_true)} pairs)")
        print(f"{'═'*55}")
        print(f"  Accuracy   : {c(acc)}{acc:.2f}%{RESET}")
        print(f"  Precision  : {c(prec)}{prec:.2f}%{RESET}")
        print(f"  Recall     : {c(rec)}{rec:.2f}%{RESET}")
        print(f"  F1 Score   : {c(f1)}{f1:.2f}%{RESET}")
        print(f"  Threshold  : probability > {threshold} = GENUINE")
        if cm.shape==(2,2):
            print(f"\n  Confusion Matrix:")
            print(f"                Predicted")
            print(f"                Genuine  Forged")
            print(f"  Actual  Gen   {cm[1][1]:>7}  {cm[1][0]:>7}")
            print(f"          Forg  {cm[0][1]:>7}  {cm[0][0]:>7}")
        print(f"{'═'*55}\n")
        return {"accuracy":round(acc,2),"precision":round(prec,2),"recall":round(rec,2),"f1":round(f1,2)}


    # ════════════════════════════════════════════════════════════════════════
    #  CLI
    # ════════════════════════════════════════════════════════════════════════
    def build_parser():
        p = argparse.ArgumentParser(prog="predict.py",
                                    description="AuthentiSign – Contrastive Siamese Verification")
        mode = p.add_mutually_exclusive_group(required=True)
        mode.add_argument("--original",  metavar="PATH")
        mode.add_argument("--batch",     metavar="CSV")
        mode.add_argument("--evaluate",  metavar="DATASET")
        p.add_argument("--test",      metavar="PATH")
        p.add_argument("--model",     metavar="PATH",  default=None)
        p.add_argument("--threshold", metavar="FLOAT", type=float, default=DECISION_THRESHOLD)
        p.add_argument("--output",    metavar="PATH",  default=None)
        p.add_argument("--num-pairs", metavar="N",     type=int,   default=100)
        p.add_argument("--json",      action="store_true")
        p.add_argument("--quiet",     action="store_true")
        return p


    def main():
        args = build_parser().parse_args()
        print(f"\n{CYAN}{BOLD}AuthentiSign – Signature Verification{RESET}")
        print(f"{'─'*55}")
        print("  Loading model …")
        try:
            m = load_model(args.model)
        except FileNotFoundError as e:
            print(f"{RED}{e}{RESET}"); sys.exit(1)

        if args.original:
            if not args.test: build_parser().error("--test required with --original")
            try:
                r = predict_pair(m, args.original, args.test, args.threshold)
            except ValueError as e:
                print(f"{RED}{e}{RESET}"); sys.exit(1)
            if args.json: print(json.dumps(r, indent=2))
            else:         print_result(r, not args.quiet)
            sys.exit(0 if r["status"]=="GENUINE" else 1)

        elif args.batch:
            if not Path(args.batch).exists():
                print(f"{RED}CSV not found: {args.batch}{RESET}"); sys.exit(1)
            batch_predict(m, args.batch, args.output, args.threshold)

        elif args.evaluate:
            evaluate_dataset(m, args.evaluate, args.num_pairs, args.threshold)


    # ════════════════════════════════════════════════════════════════════════
    #  Module API
    # ════════════════════════════════════════════════════════════════════════
    _cached_model = None

    def verify(original_path, test_path, model_path=None, threshold=DECISION_THRESHOLD):
        """
        Import and call directly:
            from predict import verify
            r = verify("original.png", "test.png")
            print(r["status"])    # "GENUINE" or "FORGED"
            print(r["distance"])  # e.g. 0.12
            print(r["confidence"])
        """
        global _cached_model
        if _cached_model is None:
            _cached_model = load_model(model_path)
        return predict_pair(_cached_model, original_path, test_path, threshold)


    if __name__ == "__main__":
        main()