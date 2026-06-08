# datasets 使用说明

把训练图片放到 `datasets/my_images/` 下。

支持两种弱提示词方式：

## 方式 1：同名 caption 文件

```text
datasets/my_images/cat_001.jpg
datasets/my_images/cat_001.txt
```

`cat_001.txt` 内容可以写：

```text
cat sitting on grass
```

## 方式 2：按文件夹名作为标签

```text
datasets/my_images/cat/001.jpg
datasets/my_images/dog/001.jpg
```

没有同名 `.txt` 时，程序会用父目录名 `cat`、`dog` 作为弱提示词。
