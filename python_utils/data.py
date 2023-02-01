import json
import gzip


def compress_url_mode(url, mode):
    if not url.endswith(".gz"):
        url += ".gz"

    if "b" not in mode:
        if mode == "w":
            mode = "wb"
        elif mode == "a":
            mode = "ab"
        elif mode == "r":
            mode = "rb"

    return url, mode


def write_json(url, data, mode="w", compress=False):
    if compress:
        url, mode = compress_url_mode(url, mode)
        f = gzip.open(url, mode)  # Write in write/append mode
    else:
        f = open(url, mode)  # Write in write/append mode
    out_data = json.dumps(data)
    if compress:
        out_data = out_data.encode()
    f.write(out_data)
    f.close()


def read_json(url, compress=False):
    if compress:
        url, mode = compress_url_mode(url, "rb")
        f = gzip.open(url, mode)
    else:
        f = open(url, "r")
    out_data = json.loads(f.read())
    f.close()
    return out_data


# def read_json(url):
#     data = []
#     with open(url) as fl:
#         data.append(json.loads(fl.read()))
#     return data


def read(file_path):
    if file_path[-3:] == ".gz":
        with gzip.open(file_path, 'rb') as f:
            lines = [line.strip() for line in f]
    else:
        with open(file_path) as f:
            lines = [line.strip() for line in f]
    return lines


def write(file_path, data, mode="w"):
    with open(file_path, mode) as f:
        if type(data) == list:
            for line in data:
                f.write(str(line))
                f.write('\n')
        else:
            f.write(str(data))
            f.write('\n')


def read_json_dump(url):
    data = []
    with open(url) as fl:
        for line in fl.readlines():
            data.append(json.loads(line))
    return data


def write_json_dump(url, data, mode="w"):
    f = open(url, mode)  # Write in write/append mode
    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        f.write(d)
        f.write('\n')
    f.close()


def write_dict(url, data, sep="~|~", mode="w"):
    f = open(url, mode)
    for k, v in data.items():
        f.write(f"{k}{sep}{v}")
        f.write('\n')
    f.close()


def read_dict(url, sep="~|~"):
    res = {}
    with open(url) as fl:
        for line in fl.read().splitlines():
            k, v = line.split(sep)
            res[k.strip()] = v.strip()
        fl.close()
    return res


def get_data_split(doc_ids, doc_labels, collection):
    data, labels = [], []
    for i in doc_ids:
        data.append(collection[i])
        labels.append(doc_labels[i])
    return {"ids": doc_ids, "data": data, "labels": labels}


def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

def count_file_lines(url,compress=False):
    if compress:
        url, mode = compress_url_mode(url, "rb")
        f = gzip.open(url, mode)
    else:
        f = open(url, "rb")

    c_generator = _count_generator(f.raw.read)
    # count each \n
    count = sum(buffer.count(b'\n') for buffer in c_generator)
    f.close()
    return count + 1