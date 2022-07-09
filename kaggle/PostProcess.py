import os, pickle, torch, tqdm
import numpy as np
import pandas as pd


def main():
    dir = r"F:\Dataset\Kaggle\before"
    with open(os.path.join(dir, 'idx2individual_id.pkl'), 'rb') as fp:
        idx2landmark_id = pickle.load(fp)
        landmark_id2idx = {idx2landmark_id[idx]: idx for idx in idx2landmark_id.keys()}

    df = pd.read_csv(os.path.join(dir, 'train.csv'))
    pred_mask = pd.Series(df.individual_id.unique()).map(landmark_id2idx).values

    TOP_K = 5
    CLS_TOP_K = 5
    if True:
        with torch.no_grad():
            feats = []
            for img0, img1, img3 in tqdm(query_loader):  # 672, 768, 512
                img0 = img0.cuda()
                img1 = img1.cuda()
                img3 = img3.cuda()

                feat_b7, _ = model_b7(img0)
                feat_b6, _ = model_b6(img1)
                feat_b5, _ = model_b5(img1)
                feat_b4, _ = model_b4(img1)
                feat_b3, _ = model_b3(img1)
                feat_nest101, _ = model_nest101(img1)
                feat_rex2, _ = model_rex2(img1)
                feat_b6b, _ = model_b6b(img3)
                feat_b5b, _ = model_b5b(img1)
                feat = torch.cat(
                    [feat_b7, feat_b6, feat_b5, feat_b4, feat_b3, feat_nest101, feat_rex2, feat_b6b, feat_b5b], dim=1)
                #             print(feat.shape)
                feats.append(feat.detach().cpu())
            feats = torch.cat(feats)
            feats = feats.cuda()
            feat = F.normalize(feat)

            PRODS = []
            PREDS = []
            PRODS_M = []
            PREDS_M = []
            for img0, img1, img3 in tqdm(test_loader):
                img0 = img0.cuda()
                img1 = img1.cuda()
                img3 = img3.cuda()

                probs_m = torch.zeros([4, 81313], device=device)
                feat_b7, logits_m = model_b7(img0)
                probs_m += logits_m
                feat_b6, logits_m = model_b6(img1)
                probs_m += logits_m
                feat_b5, logits_m = model_b5(img1)
                probs_m += logits_m
                feat_b4, logits_m = model_b4(img1)
                probs_m += logits_m
                feat_b3, logits_m = model_b3(img1)
                probs_m += logits_m
                feat_nest101, logits_m = model_nest101(img1)
                probs_m += logits_m
                feat_rex2, logits_m = model_rex2(img1)
                probs_m += logits_m
                feat_b6b, logits_m = model_b6b(img3)
                probs_m += logits_m
                feat_b5b, logits_m = model_b5b(img1)
                probs_m += logits_m
                feat = torch.cat(
                    [feat_b7, feat_b6, feat_b5, feat_b4, feat_b3, feat_nest101, feat_rex2, feat_b6b, feat_b5b], dim=1)
                feat = F.normalize(feat)

                probs_m = probs_m / 9
                probs_m[:, pred_mask] += 1.0
                probs_m -= 1.0

                (values, indices) = torch.topk(probs_m, CLS_TOP_K, dim=1)
                probs_m = values
                preds_m = indices
                PRODS_M.append(probs_m.detach().cpu())
                PREDS_M.append(preds_m.detach().cpu())

                distance = feat.mm(feats.t())
                (values, indices) = torch.topk(distance, TOP_K, dim=1)
                probs = values
                preds = indices
                PRODS.append(probs.detach().cpu())
                PREDS.append(preds.detach().cpu())

            PRODS = torch.cat(PRODS).numpy()
            PREDS = torch.cat(PREDS).numpy()
            PRODS_M = torch.cat(PRODS_M).numpy()
            PREDS_M = torch.cat(PREDS_M).numpy()

    # map both to landmark_id
    gallery_landmark = df['landmark_id'].values
    PREDS = gallery_landmark[PREDS]
    PREDS_M = np.vectorize(idx2landmark_id.get)(PREDS_M)

    PRODS_F = []
    PREDS_F = []
    for i in tqdm(range(PREDS.shape[0])):
        tmp = {}
        classify_dict = {PREDS_M[i, j]: PRODS_M[i, j] for j in range(CLS_TOP_K)}
        for k in range(TOP_K):
            lid = PREDS[i, k]
            tmp[lid] = tmp.get(lid, 0.) + float(PRODS[i, k]) ** 9 * classify_dict.get(lid, 1e-8) ** 10
        pred, conf = max(tmp.items(), key=lambda x: x[1])
        PREDS_F.append(pred)
        PRODS_F.append(conf)

    df_test['pred_id'] = PREDS_F
    df_test['pred_conf'] = PRODS_F

    df_sub['landmarks'] = df_test.apply(lambda row: f'{row["pred_id"]} {row["pred_conf"]}', axis=1)
    df_sub.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()