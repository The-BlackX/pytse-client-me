from pytse_client import download_option_data
options = download_option_data(symbols="all", write_to_excel=True, json_path="D:\\test\\symbols_option.json")
print(options["ضهرم2011"])