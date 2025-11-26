def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(b"hello world" == "hello world".encode("utf-8"))
print(decode_utf8_bytes_to_str_wrong("你好".encode("utf-8")))
# asdf