import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import argparse
import yaml
import gc
import shutil
from roboflow import Roboflow
from ultralytics import YOLO


class DatasetAugmentor:
    def __init__(self, input_path, output_path, num_augmentations=2):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.num_augmentations = num_augmentations

        # Создаем структуру директорий
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Настраиваем аугментации
        self.transform = A.Compose([
            # Изменения яркости и контраста
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.8
                ),
            ], p=0.5),

            # Шум
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),

            # Размытие
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.2),

            # Геометрические преобразования
            A.OneOf([
                A.Rotate(limit=15, p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.5
                ),
            ], p=0.3),

            # Масштабирование
            A.RandomScale(scale_limit=0.2, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def read_annotations(self, label_path):
        """Читает аннотации в формате YOLO"""
        boxes = []
        classes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x, y, w, h = map(float, data[1:])
                        boxes.append([x, y, w, h])
                        classes.append(class_id)
        except Exception as e:
            print(f"Ошибка чтения {label_path}: {e}")
        return np.array(boxes), np.array(classes)

    def write_annotations(self, boxes, classes, output_path):
        """Записывает аннотации в формате YOLO"""
        with open(output_path, 'w') as f:
            for box, cls in zip(boxes, classes):
                f.write(f"{cls} {' '.join(map(str, box))}\n")

    def apply_augmentation(self, image, boxes, classes):
        """Применяет аугментации к изображению и боксам"""
        if len(boxes) == 0:
            return image, boxes, classes
        transformed = self.transform(
            image=image,
            bboxes=boxes.tolist(),
            class_labels=classes.tolist()
        )
        return (transformed['image'], np.array(transformed['bboxes']), np.array(transformed['class_labels']))

    def process_split(self, split):
        """Обрабатывает один split датасета"""
        images_dir = self.input_path / split / 'images'
        labels_dir = self.input_path / split / 'labels'
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Пропуск сплита {split}: директории не существуют")
            return {'original': 0, 'augmented': 0}

        image_files = [f for f in images_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.png']]
        print(f"\nОбработка {split}")
        print(f"Найдено изображений: {len(image_files)}")

        stats = {'original': 0, 'augmented': 0}

        for img_path in tqdm(image_files, desc=f"Обработка {split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"\nПропуск поврежденного изображения: {img_path}")
                continue

            boxes, classes = self.read_annotations(label_path)

            new_name = f"aug_{split}_{stats['original']:06d}"
            new_img_path = self.output_path / split / 'images' / f"{new_name}.jpg"
            new_label_path = self.output_path / split / 'labels' / f"{new_name}.txt"

            cv2.imwrite(str(new_img_path), image)
            self.write_annotations(boxes, classes, new_label_path)
            stats['original'] += 1

            if len(boxes) > 0:
                for aug_idx in range(self.num_augmentations):
                    aug_image, aug_boxes, aug_classes = self.apply_augmentation(image, boxes, classes)

                    aug_name = f"aug_{split}_{stats['augmented']:06d}"
                    aug_img_path = self.output_path / split / 'images' / f"{aug_name}.jpg"
                    aug_label_path = self.output_path / split / 'labels' / f"{aug_name}.txt"

                    cv2.imwrite(str(aug_img_path), aug_image)
                    self.write_annotations(aug_boxes, aug_classes, aug_label_path)

                    stats['augmented'] += 1

            if (stats['original'] + stats['augmented']) % 100 == 0:
                gc.collect()

        return stats

    def augment_dataset(self, class_names=None):
        """Аугментирует весь датасет"""
        print("\n=== Начало аугментации датасета ===")
        total_stats = {}

        for split in ['train', 'val', 'test']:
            stats = self.process_split(split)
            total_stats[split] = stats

            print(f"\nСтатистика для {split}:")
            print(f"Оригинальных изображений: {stats['original']}")
            print(f"Аугментированных изображений: {stats['augmented']}")
            print(f"Всего: {stats['original'] + stats['augmented']}")

        print("\nСоздание конфигурационного файла data.yaml...")

        if class_names is None:
            class_names = ['fall', 'person']  # по умолчанию

        yaml_content = {
            'path': str(self.output_path),
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'val' / 'images'),
            'test': str(self.output_path / 'test' / 'images'),
            'nc': len(class_names),
            'names': class_names
        }

        with open(self.output_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print("\n=== Аугментация завершена ===")
        print(f"Результаты сохранены в: {self.output_path}")
        return str(self.output_path / 'data.yaml')


class YOLOPipeline:
    def __init__(self):
        self.args = None
        self.dataset_path = None
        self.data_yaml_path = None
        self.augmented_data_yaml_path = None

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Загрузка, аугментация и обучение YOLO модели")

        parser.add_argument("--mode", type=str, choices=['download', 'augment', 'train', 'full'], default='full')
        parser.add_argument("--data-source", type=str, choices=['roboflow', 'local'], default='roboflow')
        parser.add_argument("--api-key", type=str, default="YOUR_API_KEY")
        parser.add_argument("--workspace", type=str, default="YOUR_WORKSPACE")
        parser.add_argument("--project", type=str, default="YOUR_PROJECT")
        parser.add_argument("--version", type=int, default=1)
        parser.add_argument("--model-version", type=str, default="yolov8")
        parser.add_argument("--dataset-dir", type=str, default="dataset")
        parser.add_argument("--dataset-name", type=str, default=None)
        parser.add_argument("--augmented-dir", type=str, default="augmented_dataset")
        parser.add_argument("--num-aug", type=int, default=2)
        parser.add_argument("--class-names", type=str, nargs='+')
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--model-path", type=str, required=False)
        parser.add_argument("--results-dir", type=str, default="results")

        # Новый параметр device
        parser.add_argument("--device", type=str, default="cpu", help="cpu или номер GPU (например 0)")

        return parser.parse_args()

    def setup_local_dataset(self):
        if self.args.dataset_name:
            dataset_path = os.path.join(self.args.dataset_dir, self.args.dataset_name)
        else:
            dataset_path = self.args.dataset_dir

        if not os.path.exists(dataset_path):
            print(f"Ошибка: директория датасета не существует: {dataset_path}")
            return False

        required_subdirs = ['train/images', 'train/labels']
        for subdir in required_subdirs:
            if not os.path.exists(os.path.join(dataset_path, subdir)):
                print(f"Ошибка: в датасете отсутствует {subdir}")
                return False

        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            class_names = self.args.class_names or ['fall', 'person']
            yaml_content = {
                'path': dataset_path,
                'train': os.path.join(dataset_path, 'train', 'images'),
                'val': os.path.join(dataset_path, 'val', 'images'),
                'test': os.path.join(dataset_path, 'test', 'images'),
                'nc': len(class_names),
                'names': class_names
            }
            with open(data_yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)

        self.dataset_path = dataset_path
        self.data_yaml_path = data_yaml_path
        return True

    def augment_dataset(self):
        if not self.dataset_path:
            return False

        class_names = self.args.class_names
        if not class_names and os.path.exists(self.data_yaml_path):
            with open(self.data_yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                class_names = data_yaml.get('names', ['fall', 'person'])

        augmentor = DatasetAugmentor(
            input_path=self.dataset_path,
            output_path=self.args.augmented_dir,
            num_augmentations=self.args.num_aug
        )

        self.augmented_data_yaml_path = augmentor.augment_dataset(class_names=class_names)
        return True

    def train_model(self):
        data_yaml_path = self.augmented_data_yaml_path or self.data_yaml_path
        if not data_yaml_path:
            return False

        model = YOLO(self.args.model_path)
        results = model.train(
            data=data_yaml_path,
            epochs=self.args.epochs,
            batch=self.args.batch_size,
            imgsz=640,
            device=self.args.device,  # ВАЖНО!
            optimizer="AdamW",
            lr0=0.0001,
            weight_decay=0.0005,
            project=self.args.results_dir
        )

        print("Обучение завершено!")
        return True

    def run(self):
        self.args = self.parse_arguments()

        os.makedirs(self.args.dataset_dir, exist_ok=True)
        os.makedirs(self.args.augmented_dir, exist_ok=True)
        os.makedirs(self.args.results_dir, exist_ok=True)

        success = True

        if self.args.data_source == "local":
            success = self.setup_local_dataset()

        if success and self.args.mode in ['augment', 'full']:
            success = self.augment_dataset()

        if success and self.args.mode in ['train', 'full']:
            success = self.train_model()


def main():
    pipeline = YOLOPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
