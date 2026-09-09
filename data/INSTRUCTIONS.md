# Your handwritten images

The assignment requires your own handwriting. Images are intentionally not supplied.

1. Write `0123456789` twice, in two separate rows on clean white paper. Save a straight, tightly cropped photo or scan as `train_digits.png` in this folder.
2. Write the digits again on a separate sheet (one row `0123456789`). Save it as `test_digits.png` here.
3. Use dark strokes, even lighting, no ruled lines, shadows or page borders. Leave clear blank gaps between digits and rows. Keep each digit's strokes connected across its width. Do not split or touch digits.
4. Run `main/lab4_rbf.m`, then `additional/lab4_mlp.m` in MATLAB. Both read these same images.
5. Inspect the segmentation figures: each numbered tile must contain exactly the digit named in its title. Fix the image if segmentation is wrong, even if the total count is correct.

The simple extractor thresholds the image, separates rows and digits using blank gaps, resizes each digit to 70x50 pixels, and averages 35 blocks of 10x10 pixels. It is designed for clean, separated rows, not arbitrary photographs.

For more examples, add complete rows and edit `trainRows` or `testRows` in `shared/load_digits.m`. Keep every row in the same order. Do not reuse training images as test images. A test set of 10 digits is very small: one mistake changes accuracy by 10 percentage points.
