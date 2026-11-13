from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(r"D:\traffic-sign-demo\Outline_bao_cao_du_an_10_trang.docx")
NAVY = "1F4E79"
BLUE = "2F75B5"
PALE_BLUE = "EAF2F8"
GRAY = "5B6573"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, size=None, bold=None, color=None, italic=None):
    run.font.name = "Arial"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Arial")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if italic is not None:
        run.italic = italic


def set_cell_text(cell, text, *, bold=False, color=None, size=9.2):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_field(paragraph, field_code):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = field_code
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t")
    result.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instruction)
    run._r.append(separate)
    run._r.append(result)
    run._r.append(end)
    set_run_font(run, size=8, color=GRAY)


def add_labeled_line(doc, label, text, style=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(label)
    set_run_font(r, size=10.2, bold=True, color=NAVY)
    r = p.add_run(text)
    set_run_font(r, size=10.2, color="222222")
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.23)
    p.paragraph_format.first_line_indent = Inches(-0.14)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    r = p.add_run(text)
    set_run_font(r, size=10.1, color="222222")
    return p


def add_page_section(doc, page, title, content, visual):
    h = doc.add_paragraph(style="Heading 1")
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    r = h.add_run(f"Trang {page}  |  {title}")
    set_run_font(r, size=13.3, bold=True, color=NAVY)
    add_labeled_line(doc, "Nội dung chính. ", content)
    add_labeled_line(doc, "Hình ảnh hoặc bảng nên có. ", visual)


doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.27)
section.page_height = Inches(11.69)
section.top_margin = Inches(0.68)
section.bottom_margin = Inches(0.68)
section.left_margin = Inches(0.78)
section.right_margin = Inches(0.78)
section.header_distance = Inches(0.32)
section.footer_distance = Inches(0.32)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Arial"
normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
normal.font.size = Pt(10.2)
normal.font.color.rgb = RGBColor.from_string("222222")
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.08

for style_name, size in (("Heading 1", 13.3), ("Heading 2", 11.3)):
    style = styles[style_name]
    style.font.name = "Arial"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string(NAVY)

title_style = styles["Title"]
title_style.font.name = "Arial"
title_style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
title_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
title_style.font.size = Pt(23)
title_style.font.bold = True
title_style.font.color.rgb = RGBColor.from_string("111111")
title_style.paragraph_format.space_after = Pt(6)

header = section.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = header.add_run("ĐỀ CƯƠNG BÁO CÁO DỰ ÁN")
set_run_font(hr, size=8, bold=True, color=GRAY)

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
fr = footer.add_run("Outline 10 trang  •  ")
set_run_font(fr, size=8, color=GRAY)
add_field(footer, "PAGE")

p = doc.add_paragraph(style="Title")
p.paragraph_format.keep_with_next = True
p.add_run("Outline báo cáo dự án\nnhận diện biển báo giao thông")
for run in p.runs:
    set_run_font(run, size=23, bold=True, color="111111")

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(9)
r = p.add_run("Đề cương nội dung theo cấu trúc 10 trang")
set_run_font(r, size=12.2, color=BLUE)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run(
    "Đề cương này dành cho báo cáo dự án nhận diện bốn loại biển báo cấm bằng YOLO, "
    "kết hợp giao diện Tkinter và thực nghiệm xử lý ảnh OpenCV. Phương án tính trang gồm "
    "trang bìa; nếu yêu cầu 10 trang nội dung, chuyển trang bìa ra ngoài phần đếm."
)
set_run_font(r, size=10.4, color="333333")

add_page_section(
    doc, 1, "Trang bìa",
    "Ghi tên trường, khoa, môn học, tên đề tài, giảng viên hướng dẫn, thành viên và vai trò, lớp và thời gian thực hiện.",
    "Có thể dùng logo trường hoặc ảnh minh họa các biển báo trong phạm vi đề tài."
)
add_page_section(
    doc, 2, "Tóm tắt và mục lục",
    "Tóm tắt mục tiêu, dữ liệu, phương pháp YOLO, giao diện Tkinter và kết quả nổi bật. Thêm từ khóa như YOLO, nhận diện biển báo, OpenCV và Tkinter; đặt mục lục ngắn ở phần còn lại của trang.",
    "Sơ đồ tổng quan dự án hoặc danh sách từ khóa."
)
add_page_section(
    doc, 3, "Giới thiệu đề tài",
    "Nêu vấn đề nhận diện biển báo, lý do chọn đề tài, mục tiêu, phạm vi và ứng dụng dự kiến. Xác định bốn lớp: Cấm đỗ xe, Cấm dừng và đỗ xe, Cấm rẽ trái, Cấm rẽ phải.",
    "Ảnh minh họa bốn lớp biển báo và bảng mục tiêu/phạm vi."
)
add_page_section(
    doc, 4, "Cơ sở lý thuyết",
    "Giới thiệu ngắn về xử lý ảnh, học sâu và YOLO. Giải thích detection, bounding box, confidence, Precision, Recall và mAP. Tóm tắt cách OpenCV truyền thống dùng màu và contour để tìm vùng có thể là biển báo.",
    "Sơ đồ ảnh đầu vào → YOLO → lớp, confidence và tọa độ; bảng giải thích các chỉ số."
)
add_page_section(
    doc, 5, "Dữ liệu và tiền xử lý",
    "Mô tả nguồn ảnh, định dạng, quy trình gán nhãn YOLO và cách chia tập. Dự án có 800 ảnh: 560 train, 120 validation và 120 test; mỗi lớp có tổng cộng 200 đối tượng.",
    "Biểu đồ chia tập và một ảnh mẫu có bounding box. Đối chiếu số liệu với docs/data_summary.csv."
)
add_page_section(
    doc, 6, "Phương pháp và cấu hình mô hình",
    "Mô tả pipeline dữ liệu → huấn luyện YOLO → best.pt → dự đoán ảnh mới. Ghi cấu hình đã chạy, gồm kích thước ảnh 416; lấy epoch, batch, thiết bị và thời gian huấn luyện từ log thực tế, không tự ước lượng.",
    "Sơ đồ pipeline huấn luyện và suy luận; bảng cấu hình chạy mô hình."
)
add_page_section(
    doc, 7, "Kết quả đánh giá mô hình",
    "Trên tập test 120 ảnh, kết quả chung là Precision 0,9907; Recall 0,9944; mAP50 0,995; mAP50–95 0,877. Nhận xét theo lớp và thảo luận chỉ số thấp hơn, không suy rộng kết quả ngoài tập test.",
    "Bảng chỉ số chung và theo lớp lấy từ docs/test_metrics.csv; thêm ảnh dự đoán đúng và sai nếu có."
)

table = doc.add_table(rows=1, cols=5)
table.autofit = False
table.style = "Table Grid"
widths = [1.65, 1.18, 1.18, 1.05, 1.12]
headers = ["Lớp", "Precision", "Recall", "mAP50", "mAP50–95"]
for i, (cell, text, width) in enumerate(zip(table.rows[0].cells, headers, widths)):
    cell.width = Inches(width)
    set_cell_shading(cell, NAVY)
    set_cell_margins(cell)
    set_cell_text(cell, text, bold=True, color="FFFFFF", size=8.7)
set_repeat_table_header(table.rows[0])
rows = [
    ("Cấm đỗ xe", "0,9966", "1,0000", "0,9950", "0,8411"),
    ("Cấm dừng và đỗ xe", "0,9995", "1,0000", "0,9950", "0,7675"),
    ("Cấm rẽ trái", "1,0000", "0,9777", "0,9950", "0,9515"),
    ("Cấm rẽ phải", "0,9668", "1,0000", "0,9950", "0,9480"),
]
for idx, row_values in enumerate(rows):
    cells = table.add_row().cells
    for j, (cell, text, width) in enumerate(zip(cells, row_values, widths)):
        cell.width = Inches(width)
        set_cell_margins(cell)
        if idx % 2 == 1:
            set_cell_shading(cell, PALE_BLUE)
        set_cell_text(cell, text, size=8.7)

add_page_section(
    doc, 8, "Thực nghiệm OpenCV và so sánh",
    "Mô tả các ca ảnh đủ sáng, thiếu sáng, biển nhỏ ở xa, biển lớn và nền phức tạp. So sánh khả năng phát hiện và các lỗi như bỏ sót hoặc dương tính giả; lấy IoU và số liệu từ báo cáo/log thí nghiệm của nhóm.",
    "Bảng so sánh OpenCV với YOLO theo từng tình huống và ảnh kết quả tiêu biểu."
)
add_page_section(
    doc, 9, "Ứng dụng Tkinter và kiểm thử",
    "Trình bày thao tác chọn ảnh, chỉnh confidence, nhận diện, xem khung và bảng lớp/vị trí, rồi lưu ảnh kết quả. Mô tả tích hợp load_model/predict_image và suy luận nền. Nêu các ca đã kiểm tra: ảnh có tên tiếng Việt, ảnh trắng không có detection và lưu PNG.",
    "Ảnh chụp giao diện trước/sau nhận diện và lưu kết quả. Ghi rõ ảnh tổng hợp chỉ kiểm tra luồng giao diện, không dùng làm bằng chứng độ chính xác mô hình."
)
add_page_section(
    doc, 10, "Kết luận và tài liệu tham khảo",
    "Tóm tắt mức độ đạt mục tiêu; nêu hạn chế như phạm vi bốn lớp, quy mô dữ liệu và nhu cầu kiểm thử thêm trên ảnh thực tế. Đề xuất mở rộng dữ liệu, kiểm thử điều kiện ánh sáng/góc chụp và tối ưu triển khai.",
    "Liệt kê tài liệu YOLO/OpenCV, nguồn dữ liệu, log đánh giá và tài liệu dự án."
)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(0)
r = p.add_run("Ghi chú biên soạn: ")
set_run_font(r, size=9.3, bold=True, color=NAVY)
r = p.add_run(
    "Các chỉ số ở trang 7 lấy từ kết quả hiện có của dự án. Với thông số huấn luyện hoặc kết quả OpenCV, "
    "hãy điền theo log và báo cáo gốc của nhóm."
)
set_run_font(r, size=9.3, color=GRAY)

doc.core_properties.title = "Outline báo cáo dự án nhận diện biển báo giao thông"
doc.core_properties.subject = "Đề cương báo cáo dự án 10 trang"
doc.core_properties.author = ""
doc.save(OUT)
print(OUT)
