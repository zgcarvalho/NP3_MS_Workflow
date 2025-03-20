import requests, os
from tqdm import tqdm

# retrieve a file from an url and save to local path - download with progress bar
# url: http to retrieve
# fname: path with file name, where to store the downloaded file
# chunk_size: download chunk size in Mbytes
# code from https://stackoverflow.com/questions/1308542/how-to-catch-404-error-in-urllib-urlretrieve
def download_file(url: str, fname: str, chunk_size:int=1):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        chunk_size = chunk_size * 10 ** 6
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    code = resp.status_code
    content_lenght = resp.headers['Content-length']
    if code == 200:
        if int(content_lenght) < 128883728:
            print("ERROR - The file was retrieved incomplete and will be removed")
            # remove file
            os.remove(fname) if os.path.exists(fname) else None
            sys.exit(1)
        print("File downloaded successfully!")
    elif code == 404:
        print("ERROR - The file was not found. HTTP status code 404")
        sys.exit(1)
    elif code > 200 and code < 400:
        print("Warning, check if file was downloaded without errors. HTTP status code was:", code)


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) >= 3:
        # print(sys.argv)
        url = sys.argv[1]
        file_name = sys.argv[2]
    else:
        print("Error: Two arguments must be supplied to download a file from an url and save to a local file path:\n",
            " 1 - url: The url http ou https where the file is located in the web to be downloaded from;\n",
            " 2 - file_name: The path with file name to where to store the downloaded content.\n")
        sys.exit(1)
    # call download file
    download_file(url=url, fname=file_name)
