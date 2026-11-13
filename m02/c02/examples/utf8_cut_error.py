note = "Giao trước 17 giờ"
data = note.encode("utf-8")
print(f"{len(note)} characters, {len(data)} bytes")

# Cột ghi chú của hệ thống cũ chỉ giữ 10 byte đầu
cut = data[:10]
print("first 10 bytes:", cut.hex(" "))
print(cut.decode("utf-8"))
