# esun_sar_baseline
- baseline code for competition [你說可疑不可疑？－疑似洗錢交易預測](https://tbrain.trendmicro.com.tw/Competitions/Details/24)
- Public LB Score: 0.03413

## 簡述：
- 把每種 data (ccba, cdtx, dp, remit, custinfo) 都先經過各別的 encoder 變成一個 embedding
  - encoder原始碼請見: [link](https://github.com/AxotZero/esun_sar_baseline/blob/main/src/model/modules/feature_embedder.py)
- 把顧客上述的 embedding 依據資料時間排序
- 丟到一個 Transformer 來進行預測 (Maxlen=1024, DebertaTransformerEncoder)
- loss_func:
  ```python3=
  def cost_sensetive_bce_loss(output, target, epsilon=1e-7, w_tp=99, w_tn=0, w_fp=1, w_fn=99):

    fn = w_fn * torch.mean(target * torch.log(output+epsilon))
    tp = w_tp * torch.mean(target * torch.log((1-output)+epsilon))
    fp = w_fp * torch.mean((1-target) * torch.log((1-output)+epsilon))
    tn = w_tn * torch.mean((1-target) * torch.log(output+epsilon))
    return -(fn+tp+fp+tn)
  ```
