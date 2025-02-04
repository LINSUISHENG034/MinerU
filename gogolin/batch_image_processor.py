import os
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.data.read_api import read_local_images
from magic_pdf.libs.config_reader import get_local_models_dir
from magic_pdf.libs.clean_memory import clean_memory


class BatchImageProcessor:
    """批量图片处理器
    
    用于批量处理图片文件，将图片内容转换为markdown文件。
    支持多种图片格式，优化的OCR参数配置，以及详细的处理日志。
    """

    def __init__(
        self,
        output_dir: str = "output",
        vram_size: str = "8",
        ocr_thresh: str = "0.2",
        ocr_batch_num: str = "6",
        supported_formats: List[str] = None
    ):
        """初始化批量图片处理器
        
        Args:
            output_dir (str): 输出目录的根路径，默认为"output"
            vram_size (str): 虚拟显存大小（GB），默认为"8"
            ocr_thresh (str): OCR检测阈值，越低越敏感，默认为"0.2"
            ocr_batch_num (str): OCR批处理数量，默认为"6"
            supported_formats (List[str]): 支持的图片格式列表，默认为[".png", ".jpg", ".jpeg"]
        """
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.image_dir_name = os.path.basename(self.image_dir)
        
        # 创建输出目录
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 设置支持的图片格式
        self.supported_formats = supported_formats or [".png", ".jpg", ".jpeg"]
        
        # 初始化环境变量
        self._init_environment(vram_size, ocr_thresh, ocr_batch_num)
        
        # 初始化文件写入器
        self.image_writer = FileBasedDataWriter(self.image_dir)
        self.md_writer = FileBasedDataWriter(self.output_dir)

    def _init_environment(self, vram_size: str, ocr_thresh: str, ocr_batch_num: str):
        """初始化环境变量
        
        Args:
            vram_size (str): 虚拟显存大小
            ocr_thresh (str): OCR检测阈值
            ocr_batch_num (str): OCR批处理数量
        """
        # 设置模型路径
        models_dir = get_local_models_dir()
        if models_dir:
            normalized_path = str(Path(models_dir).resolve())
            os.environ['LOCAL_MODELS_DIR'] = normalized_path
        
        # 设置虚拟显存大小
        os.environ['VIRTUAL_VRAM_SIZE'] = vram_size
        
        # 设置OCR参数
        os.environ['OCR_DET_DB_THRESH'] = ocr_thresh  # 检测阈值，越低越敏感
        os.environ['OCR_REC_BATCH_NUM'] = ocr_batch_num  # 批处理数量

    def process_directory(
        self,
        input_directory: str,
        lang: str = "ch",
        show_log: bool = True,
        layout_model: str = "doclayout_yolo",
        formula_enable: bool = False
    ) -> Tuple[int, int, int]:
        """处理指定目录下的所有图片
        
        Args:
            input_directory (str): 输入目录路径
            lang (str): OCR语言，默认为"ch"（中文）
            show_log (bool): 是否显示详细日志，默认为True
            layout_model (str): 布局检测模型，默认为"doclayout_yolo"
            formula_enable (bool): 是否启用公式检测，默认为False
            
        Returns:
            Tuple[int, int, int]: 返回(总图片数, 处理成功数, 处理失败数)
        """
        # 获取绝对路径
        abs_input_dir = os.path.abspath(input_directory)
        logger.info(f"Processing directory: {abs_input_dir}\n")
        
        # 验证目录
        if not os.path.exists(abs_input_dir) or not os.path.isdir(abs_input_dir):
            logger.error(f"Directory does not exist: {abs_input_dir}")
            raise ValueError(f"Invalid directory: {abs_input_dir}")
        
        # 收集图片文件
        image_files = self._collect_image_files(abs_input_dir)
        logger.info(f"\nFound {len(image_files)} image files")
        
        # 加载数据集
        dss, ds_paths = self._load_datasets(image_files)
        if not dss:
            logger.error("\nNo images could be loaded. Please check if:")
            logger.error(f"1. Directory exists: {abs_input_dir}")
            logger.error("2. Files have correct permissions")
            logger.error("3. Files are valid images")
            raise ValueError("No valid images could be loaded")
        
        # 处理图片
        return self._process_images(
            dss,
            ds_paths,
            lang=lang,
            show_log=show_log,
            layout_model=layout_model,
            formula_enable=formula_enable
        )

    def _collect_image_files(self, directory: str) -> List[Tuple[str, str]]:
        """收集目录中的图片文件
        
        Args:
            directory (str): 目录路径
            
        Returns:
            List[Tuple[str, str]]: 返回(文件路径, 文件名)列表
        """
        image_files = []
        logger.info("Files in directory:")
        for file in os.listdir(directory):
            suffix = Path(file).suffix
            logger.info(f"- {file} (suffix: {suffix})")
            if suffix.lower() in self.supported_formats:
                file_path = os.path.join(directory, file)
                image_files.append((file_path, file))
        return image_files

    def _load_datasets(
        self,
        image_files: List[Tuple[str, str]]
    ) -> Tuple[List, List[str]]:
        """加载图片数据集
        
        Args:
            image_files (List[Tuple[str, str]]): (文件路径, 文件名)列表
            
        Returns:
            Tuple[List, List[str]]: 返回(数据集列表, 文件名列表)
        """
        dss = []
        ds_paths = []
        for image_path, image_name in image_files:
            try:
                ds = read_local_images(image_path, suffixes=self.supported_formats)
                if ds:
                    dss.extend(ds)
                    ds_paths.extend([image_name] * len(ds))
            except Exception as e:
                logger.error(f"Error reading image {image_path}: {str(e)}")
        logger.info(f"Successfully loaded {len(dss)} images\n")
        return dss, ds_paths

    def _process_images(
        self,
        dss: List,
        ds_paths: List[str],
        lang: str,
        show_log: bool,
        layout_model: str,
        formula_enable: bool
    ) -> Tuple[int, int, int]:
        """处理图片数据集
        
        Args:
            dss (List): 数据集列表
            ds_paths (List[str]): 文件名列表
            lang (str): OCR语言
            show_log (bool): 是否显示详细日志
            layout_model (str): 布局检测模型
            formula_enable (bool): 是否启用公式检测
            
        Returns:
            Tuple[int, int, int]: 返回(总图片数, 处理成功数, 处理失败数)
        """
        processed = 0
        failed = 0
        total = len(dss)

        for i, (ds, image_name) in enumerate(zip(dss, ds_paths)):
            try:
                logger.info(f"Processing image {i+1}/{total}: {image_name}")
                
                # 提取文件名（不含扩展名）
                filename = os.path.splitext(image_name)[0]
                
                # 处理图片
                ds.apply(
                    doc_analyze,
                    ocr=True,
                    lang=lang,
                    show_log=show_log,
                    layout_model=layout_model,
                    formula_enable=formula_enable
                ).pipe_ocr_mode(self.image_writer).dump_md(
                    self.md_writer, f"{filename}.md", self.image_dir_name
                )
                
                logger.info(f"Successfully processed: {filename}")
                processed += 1
                
                # 清理内存
                clean_memory()
                
            except Exception as e:
                logger.error(f"Error processing image {image_name}: {str(e)}")
                failed += 1
                continue

        logger.info(f"\nProcessing complete:")
        logger.info(f"Total images: {total}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Failed: {failed}")
        
        return total, processed, failed