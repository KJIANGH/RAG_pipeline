from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
import re

def extract_docx_metadata(doc_elements):
    metadata = []
    current_section = None

    for el in doc_elements:
        text = el.text.strip()

        # 提取编号 (e.g., 1, 1.1, 1.1.1, a, b)
        match = re.match(r"^(\d+(\.\d+)*|[a-zA-Z])[\.\)]?\s+(.*)", text)
        if match:
            section_number = match.group(1)
            content = match.group(3)

            # 检查是否有粗体
            term = None
            if el.metadata and el.metadata.bold:
                term = content.split()[0]  # 简化：取第一个词为 term
            metadata.append({
                "section": section_number,
                "term": term,
                "text": content,
            })
            current_section = section_number

        elif el.metadata and el.metadata.bold and current_section:
            term = text.split()[0]
            metadata.append({
                "section": current_section,
                "term": term,
                "text": text,
            })

    return metadata


def extract_pdf_metadata(doc_elements):
    metadata = []

    for el in doc_elements:
        text = el.text.strip()

        # 情况 A：粗体 + means
        if el.metadata and el.metadata.bold and " means" in text:
            term_match = re.match(r"^([A-Z][A-Za-z0-9\s]+?)\s+means", text)
            if term_match:
                term = term_match.group(1).strip()
                metadata.append({
                    "section": None,
                    "term": term,
                    "text": text,
                })

        # 情况 B：编号 + 无粗体
        elif re.match(r"^(\d+(\.\d+)*|[a-zA-Z])[\.\)]?\s+", text):
            section_number = re.match(r"^(\d+(\.\d+)*|[a-zA-Z])[\.\)]?", text).group(1)
            metadata.append({
                "section": section_number,
                "term": None,
                "text": text,
            })

    return metadata


def extract_metadata(element, file_path):
    if file_path.lower().endswith('.docx'):
        return extract_docx_metadata(element)
    elif file_path.lower().endswith('.pdf'):
        return extract_pdf_metadata(element)
    else:
        raise ValueError("Unsupported file format for metadata extraction")
