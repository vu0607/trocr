import os

from PIL import Image, ImageFont, ImageDraw


class SaveTestAnnotation:
    def __init__(self, font_path, size):
        self.font = ImageFont.truetype(font_path, size)

    def __call__(self, dir_path, file_name, out_path, pred, label):
        raw_img = Image.open(os.path.join(dir_path, file_name))

        self.raw_size = raw_img.size
        self.pred_size = self.font.getsize(pred)
        self.label_size = self.font.getsize(label)

        pred_img = self._draw_text(self.pred_size, pred)
        label_img = self._draw_text(self.label_size, label)

        dst = Image.new('RGB', (self._get_width(), self._get_height()), color=(0, 0, 0))
        dst.paste(raw_img, (0, 0))
        dst.paste(label_img, (0, raw_img.height))
        dst.paste(pred_img, (0, raw_img.height + label_img.height))

        dst.save(os.path.join(out_path, file_name))

        return dst

    def _draw_text(self, size, text):
        img = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        draw_pred = ImageDraw.Draw(img)
        draw_pred.text((0, 0), text, (0, 0, 0), font=self.font)
        return img

    def _get_width(self):
        return max(self.raw_size[0], self.pred_size[0], self.label_size[0])

    def _get_height(self):
        return sum([self.raw_size[1], self.pred_size[1], self.label_size[1]])
