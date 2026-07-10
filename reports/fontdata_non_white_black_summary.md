# fontdata non-white-black image report

- scanned_root: `fontdata_example/font/train/new`
- target_folder_per_font: `images_white_bg_mask_denoised`
- rule: mark as non white-background black-text if `(white_ratio < 0.70 and dark_ratio > 0.25) or (white_ratio < 0.50 and border_mean < 220)`
- note: edge-touching/cropped but still white-background black-text images are intentionally excluded from this stricter list.
- invalid_count: `607`

## invalid_by_font

| font | count |
|---|---:|
| `OuyxBF` | 101 |
| `OuyxJT` | 79 |
| `SushiBF` | 67 |
| `SushiJT` | 40 |
| `YanzqBF` | 3 |
| `YanzqJT` | 13 |
| `ZhaomfBF` | 184 |
| `ZhaomfJT` | 120 |

## first_examples

| font | filename | white_ratio | dark_ratio | border_mean |
|---|---|---:|---:|---:|
| `OuyxBF` | `乃.png` | 0.126 | 0.803 | 139.6 |
| `OuyxBF` | `人.png` | 0.387 | 0.567 | 174.5 |
| `OuyxBF` | `伐.png` | 0.056 | 0.840 | 62.5 |
| `OuyxBF` | `位.png` | 0.099 | 0.789 | 136.1 |
| `OuyxBF` | `光.png` | 0.137 | 0.757 | 141.6 |
| `OuyxBF` | `冬.png` | 0.156 | 0.768 | 140.0 |
| `OuyxBF` | `出.png` | 0.100 | 0.807 | 134.1 |
| `OuyxBF` | `列.png` | 0.054 | 0.851 | 128.4 |
| `OuyxBF` | `制.png` | 0.123 | 0.762 | 137.7 |
| `OuyxBF` | `召.png` | 0.094 | 0.816 | 37.2 |
| `OuyxBF` | `吊.png` | 0.156 | 0.765 | 141.4 |
| `OuyxBF` | `唐.png` | 0.098 | 0.787 | 133.8 |
| `OuyxBF` | `國.png` | 0.229 | 0.619 | 142.2 |
| `OuyxBF` | `地.png` | 0.248 | 0.660 | 151.7 |
| `OuyxBF` | `坐.png` | 0.086 | 0.828 | 131.8 |
| `OuyxBF` | `夜.png` | 0.106 | 0.780 | 138.4 |
| `OuyxBF` | `天.png` | 0.177 | 0.757 | 142.8 |
| `OuyxBF` | `始.png` | 0.166 | 0.688 | 137.6 |
| `OuyxBF` | `字.png` | 0.136 | 0.765 | 138.7 |
| `OuyxBF` | `宇.png` | 0.113 | 0.802 | 133.8 |
| `OuyxBF` | `官.png` | 0.173 | 0.712 | 140.0 |
| `OuyxBF` | `宙.png` | 0.136 | 0.754 | 134.2 |
| `OuyxBF` | `宿.png` | 0.063 | 0.817 | 17.8 |
| `OuyxBF` | `寒.png` | 0.126 | 0.757 | 136.9 |
| `OuyxBF` | `岡.png` | 0.161 | 0.718 | 144.9 |
| `OuyxBF` | `崐.png` | 0.068 | 0.793 | 133.7 |
| `OuyxBF` | `巨.png` | 0.102 | 0.789 | 137.9 |
| `OuyxBF` | `帝.png` | 0.059 | 0.829 | 126.8 |
| `OuyxBF` | `師.png` | 0.201 | 0.678 | 145.3 |
| `OuyxBF` | `張.png` | 0.222 | 0.675 | 151.8 |
| `OuyxBF` | `往.png` | 0.167 | 0.760 | 142.1 |
| `OuyxBF` | `律.png` | 0.175 | 0.735 | 148.1 |
| `OuyxBF` | `成.png` | 0.062 | 0.832 | 131.6 |
| `OuyxBF` | `拱.png` | 0.128 | 0.751 | 130.7 |
| `OuyxBF` | `推.png` | 0.176 | 0.713 | 145.7 |
| `OuyxBF` | `收.png` | 0.187 | 0.732 | 144.7 |
| `OuyxBF` | `文.png` | 0.216 | 0.714 | 148.1 |
| `OuyxBF` | `日.png` | 0.226 | 0.723 | 148.6 |
| `OuyxBF` | `昃.png` | 0.165 | 0.753 | 141.4 |
| `OuyxBF` | `暑.png` | 0.113 | 0.783 | 135.5 |
| `OuyxBF` | `月.png` | 0.102 | 0.817 | 133.8 |
| `OuyxBF` | `有.png` | 0.100 | 0.803 | 126.9 |
| `OuyxBF` | `服.png` | 0.112 | 0.760 | 135.5 |
| `OuyxBF` | `朝.png` | 0.126 | 0.752 | 134.1 |
| `OuyxBF` | `李.png` | 0.250 | 0.656 | 156.4 |
| `OuyxBF` | `来.png` | 0.114 | 0.794 | 133.4 |
| `OuyxBF` | `歲.png` | 0.114 | 0.796 | 137.9 |
| `OuyxBF` | `殷.png` | 0.056 | 0.814 | 91.4 |
| `OuyxBF` | `民.png` | 0.189 | 0.703 | 139.4 |
| `OuyxBF` | `水.png` | 0.133 | 0.795 | 143.7 |
| `OuyxBF` | `河.png` | 0.082 | 0.804 | 135.0 |
| `OuyxBF` | `洪.png` | 0.135 | 0.766 | 138.5 |
| `OuyxBF` | `海.png` | 0.151 | 0.724 | 143.4 |
| `OuyxBF` | `淡.png` | 0.119 | 0.770 | 141.7 |
| `OuyxBF` | `湯.png` | 0.066 | 0.811 | 105.8 |
| `OuyxBF` | `潛.png` | 0.086 | 0.803 | 136.4 |
| `OuyxBF` | `火.png` | 0.113 | 0.823 | 138.4 |
| `OuyxBF` | `玄.png` | 0.206 | 0.731 | 142.7 |
| `OuyxBF` | `玉.png` | 0.232 | 0.705 | 151.2 |
| `OuyxBF` | `珎.png` | 0.071 | 0.816 | 128.8 |
| `OuyxBF` | `珠.png` | 0.059 | 0.825 | 131.7 |
| `OuyxBF` | `生.png` | 0.075 | 0.845 | 133.1 |
| `OuyxBF` | `皇.png` | 0.278 | 0.631 | 155.7 |
| `OuyxBF` | `盈.png` | 0.134 | 0.747 | 138.7 |
| `OuyxBF` | `秋.png` | 0.188 | 0.720 | 143.6 |
| `OuyxBF` | `稱.png` | 0.062 | 0.786 | 130.1 |
| `OuyxBF` | `结.png` | 0.117 | 0.766 | 138.7 |
| `OuyxBF` | `罪.png` | 0.208 | 0.679 | 138.9 |
| `OuyxBF` | `羽.png` | 0.107 | 0.764 | 133.3 |
| `OuyxBF` | `翔.png` | 0.072 | 0.778 | 128.4 |
| `OuyxBF` | `致.png` | 0.049 | 0.837 | 129.9 |
| `OuyxBF` | `芥.png` | 0.221 | 0.699 | 153.6 |
| `OuyxBF` | `荒.png` | 0.171 | 0.734 | 147.1 |
| `OuyxBF` | `菓.png` | 0.225 | 0.651 | 150.9 |
| `OuyxBF` | `菜.png` | 0.038 | 0.873 | 129.9 |
| `OuyxBF` | `藏.png` | 0.078 | 0.811 | 132.4 |
| `OuyxBF` | `虞.png` | 0.130 | 0.758 | 136.1 |
| `OuyxBF` | `號.png` | 0.249 | 0.618 | 157.3 |
| `OuyxBF` | `衣.png` | 0.091 | 0.811 | 134.2 |
| `OuyxBF` | `裳.png` | 0.084 | 0.767 | 130.6 |
