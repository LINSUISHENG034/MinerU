from loguru import logger
from batch_image_processor import BatchImageProcessor

def main():
    """
    使用示例：演示如何使用BatchImageProcessor处理图片
    """
    try:
        # 创建处理器实例
        # 参数说明：
        # - output_dir: 输出目录，默认为"output"
        # - vram_size: 虚拟显存大小，默认为"8"GB
        # - ocr_thresh: OCR检测阈值，默认为"0.2"（越低越敏感）
        # - ocr_batch_num: OCR批处理数量，默认为"6"
        processor = BatchImageProcessor(
            output_dir="output",
            vram_size="8",
            ocr_thresh="0.2",
            ocr_batch_num="6"
        )
        
        # 处理指定目录下的图片
        # 参数说明：
        # - input_directory: 输入目录路径
        # - lang: OCR语言，"ch"为中文
        # - show_log: 是否显示详细日志
        # - layout_model: 布局检测模型，使用"doclayout_yolo"
        # - formula_enable: 是否启用公式检测
        total, processed, failed = processor.process_directory(
            input_directory="my_test/source/image/video1",
            lang="ch",
            show_log=True,
            layout_model="doclayout_yolo",
            formula_enable=False
        )
        
        # 输出处理结果
        logger.info("\n处理结果统计：")
        logger.info(f"总图片数：{total}")
        logger.info(f"成功处理：{processed}")
        logger.info(f"处理失败：{failed}")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()