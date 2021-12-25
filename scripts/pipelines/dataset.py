from scripts.data.ticker_data import TickerData
from scripts.dataset.dataset import WindowedDataset
from scripts.utils.logger import setup_logger

logger = setup_logger('Dataset Pipeline')


def dataset_pipeline():
    ticker = 'ETH-EUR'
    label_window = 5
    model_window = 10
    preprocessing_version = 1

    logger.info(f' > Loading Data')
    ticker_data = TickerData(ticker=ticker,
                             window=label_window)

    logger.info(f' > Labeling Data')
    labeled_data = ticker_data.label_data()

    logger.info(f' > Data Preprocessing')
    preprocessed_data = ticker_data.preprocessing(preprocessing_version)
    ticker_data.preprocessed_data = ticker_data.preprocessed_data.dropna()

    logger.info(f' > Creating Dataset')
    dataset = WindowedDataset(ticker_data.preprocessed_data, window=model_window)
    logger.info(f' > Window shape: {dataset.shape()}')

    return
