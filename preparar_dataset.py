#!/usr/bin/env python3.11
"""
Script para preparar o dataset de tomates para treinamento com YOLOv8
Divide as imagens em conjuntos de treino, validação e teste
"""

import os
import shutil
import random
from pathlib import Path

def dividir_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Divide o dataset em train/val/test
    
    Args:
        source_dir: Diretório com as imagens originais
        dest_dir: Diretório de destino com estrutura YOLO
        train_ratio: Proporção para treino (padrão: 70%)
        val_ratio: Proporção para validação (padrão: 20%)
        test_ratio: Proporção para teste (padrão: 10%)
    """
    
    # Verificar se as proporções somam 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "As proporções devem somar 1.0"
    
    # Obter lista de imagens
    source_path = Path(source_dir)
    images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
    
    print(f"Total de imagens encontradas: {len(images)}")
    
    # Embaralhar aleatoriamente
    random.seed(42)  # Para reprodutibilidade
    random.shuffle(images)
    
    # Calcular índices de divisão
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Dividir em conjuntos
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"\nDivisão do dataset:")
    print(f"  Treino: {len(train_images)} imagens ({len(train_images)/total*100:.1f}%)")
    print(f"  Validação: {len(val_images)} imagens ({len(val_images)/total*100:.1f}%)")
    print(f"  Teste: {len(test_images)} imagens ({len(test_images)/total*100:.1f}%)")
    
    # Criar diretórios de destino
    dest_path = Path(dest_dir)
    for split in ['train', 'val', 'test']:
        (dest_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dest_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copiar imagens
    print("\nCopiando imagens...")
    
    def copiar_imagens(image_list, split_name):
        for img in image_list:
            dest_img = dest_path / 'images' / split_name / img.name
            shutil.copy2(img, dest_img)
    
    copiar_imagens(train_images, 'train')
    copiar_imagens(val_images, 'val')
    copiar_imagens(test_images, 'test')
    
    print("✓ Dataset dividido com sucesso!")
    
    return {
        'train': len(train_images),
        'val': len(val_images),
        'test': len(test_images)
    }

def criar_data_yaml(dest_dir, num_classes=2, class_names=None):
    """
    Cria o arquivo data.yaml necessário para o YOLOv8
    
    Args:
        dest_dir: Diretório raiz do dataset
        num_classes: Número de classes
        class_names: Lista com nomes das classes
    """
    
    if class_names is None:
        class_names = ['ripe', 'unripe']  # Padrão: maduro e verde
    
    dest_path = Path(dest_dir)
    yaml_content = f"""# Dataset de Tomates para YOLOv8
# Gerado automaticamente

# Caminhos (relativos ao arquivo data.yaml)
train: images/train
val: images/val
test: images/test

# Número de classes
nc: {num_classes}

# Nomes das classes
names: {class_names}
"""
    
    yaml_path = dest_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Arquivo data.yaml criado em: {yaml_path}")

def main():
    """Função principal"""
    
    print("=" * 60)
    print("PREPARAÇÃO DO DATASET DE TOMATES PARA YOLOV8")
    print("=" * 60)
    
    # Diretórios
    source_dir = "/home/ubuntu/dataset_tomates/images"
    dest_dir = "/home/ubuntu/dataset_yolo"
    
    # Dividir dataset
    stats = dividir_dataset(source_dir, dest_dir)
    
    # Criar arquivo data.yaml
    criar_data_yaml(dest_dir, num_classes=2, class_names=['ripe', 'unripe'])
    
    print("\n" + "=" * 60)
    print("PRÓXIMOS PASSOS:")
    print("=" * 60)
    print("1. Anotar as imagens usando Roboflow, LabelImg ou CVAT")
    print("2. Salvar as anotações no formato YOLO (.txt)")
    print("3. Colocar os arquivos .txt nos diretórios labels/train, labels/val, labels/test")
    print("4. Executar o treinamento com: yolo train data=dataset_yolo/data.yaml")
    print("=" * 60)

if __name__ == "__main__":
    main()

