class DataSource:
    CCBA = 0
    CDTX = 1
    DP = 2
    REMIT = 3
    CUSTINFO = 4


class FeatureType:
    ID = 0
    DATE = 1
    CATEGORICAL = 2
    NUMERICAL = 3
    TARGET = 4


class CCBAConfig:
    cust_id = FeatureType.ID
    lupay = FeatureType.NUMERICAL
    byymm = FeatureType.DATE
    cycam = FeatureType.NUMERICAL
    usgam = FeatureType.NUMERICAL
    clamt = FeatureType.NUMERICAL
    csamt = FeatureType.NUMERICAL
    inamt = FeatureType.NUMERICAL
    cucsm = FeatureType.NUMERICAL
    cucah = FeatureType.NUMERICAL


class CDTXConfig:
    cust_id = FeatureType.ID
    date = FeatureType.DATE
    country = FeatureType.CATEGORICAL
    amt = FeatureType.NUMERICAL
    cur_type = FeatureType.CATEGORICAL


class DPConfig:
    cust_id = FeatureType.ID
    debit_credit = FeatureType.CATEGORICAL
    tx_date = FeatureType.DATE
    tx_time = FeatureType.CATEGORICAL
    tx_type = FeatureType.CATEGORICAL
    tx_amt = FeatureType.NUMERICAL
    exchg_rate = FeatureType.NUMERICAL
    info_asset_code = FeatureType.CATEGORICAL
    txbranch = FeatureType.CATEGORICAL
    fiscTxId = FeatureType.CATEGORICAL
    cross_bank = FeatureType.CATEGORICAL
    ATM = FeatureType.CATEGORICAL


class REMITConfig:
    cust_id = FeatureType.ID
    trans_date = FeatureType.DATE
    trans_no = FeatureType.CATEGORICAL
    trade_amount_usd = FeatureType.NUMERICAL


class CUSTINFOConfig:
    cust_id = FeatureType.ID
    alert_key = FeatureType.ID
    risk_rank = FeatureType.CATEGORICAL
    occupation_code = FeatureType.CATEGORICAL
    total_asset = FeatureType.NUMERICAL
    AGE = FeatureType.CATEGORICAL
    date = FeatureType.DATE
    sar_flag = FeatureType.TARGET


CONFIG_MAP = {
    DataSource.CCBA: CCBAConfig,
    DataSource.CDTX: CDTXConfig,
    DataSource.DP: DPConfig,
    DataSource.REMIT: REMITConfig,
    DataSource.CUSTINFO: CUSTINFOConfig,
}

DATA_SOURCES = [DataSource.CCBA, DataSource.CDTX, DataSource.DP, DataSource.REMIT, DataSource.CUSTINFO]