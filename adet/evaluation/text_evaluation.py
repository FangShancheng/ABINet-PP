import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from adet.evaluation import text_eval_script
import zipfile
import pickle


class TextEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )
        
        self.voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.BATEXT.CUSTOM_DICT
        if not self.use_customer_dictionary:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # use dataset_name to decide eval_gt_path
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_totaltext.zip"
            self._word_spotting = True
        elif "icdar2015" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_icdar2015.zip"
            self._word_spotting = False
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_ctw1500.zip"
            self._word_spotting = False
        else:
            self._text_eval_gt_path = ""
        self._text_eval_confidence = cfg.MODEL.FCOS.INFERENCE_TH_TEST

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input["image_id"])
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.5):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            a = [c for c in s if ord(c) < 128]
            outa = ''
            for i in a:
                outa +=i
            return outa

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open(os.path.join(temp_dir,'temp_all_det_cors.txt'), 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0 
                        ymax = 0
                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + str(int(data[ix]['polys'][i][0])) +','+str(int(data[ix]['polys'][i][1])) +','
                        ass = de_ascii(data[ix]['rec'])
                        if len(ass)>=0: # 
                            outstr = outstr + str(round(data[ix]['score'], 3)) +',####'+ass+'\n'	
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th] 
        fres = open(os.path.join(temp_dir,'temp_all_det_cors.txt'), 'r').readlines()
        for isc in lsc:	
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')
                filename = '{:07d}.txt'.format(int(s[0]))
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    ptr = s[1].strip().split(',####')
                    score = ptr[0].split(',')[-1]
                    if float(score) < isc:
                        continue
                    cors = ','.join(e for e in ptr[0].split(',')[:-1])
                    fout.writelines(cors+',####'+ptr[1]+'\n')
        os.remove(os.path.join(temp_dir,'temp_all_det_cors.txt'))

    def sort_detection(self, temp_dir, output_dir):
        origin_file = temp_dir
        output_file = temp_dir + "_final"

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file+'/*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',####')
                rec  = ptr[1]
                cors = ptr[0].split(',')
                assert(len(cors) %2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                
                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                    
                pRing = LinearRing(pts)
                if pRing.is_ccw:
                    pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                outstr = outstr+',####' + rec
                fout.writelines(outstr+'\n')
            fout.close()
        current_path = os.getcwd()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        det_path = os.path.join(output_dir, 'det.zip')
        zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir(current_path)
        # clean temp files
        shutil.rmtree(origin_file)
        shutil.rmtree(output_file)
        return det_path
    
    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        PathManager.mkdirs(self._output_dir)

        file_path = os.path.join(self._output_dir, "text_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()
        
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        # eval text
        temp_dir = os.path.join(self._output_dir, "temp_det_results")
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path = self.sort_detection(temp_dir, self._output_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
        os.remove(result_path)

        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        for task in ("e2e_method", "det_only_method"):
            result = text_result[task]
            groups = re.match(template, result).groups()
            self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}

        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, img_id):
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        if hasattr(instances, 'beziers'):
            beziers = instances.beziers.numpy()
        else:
            beziers = instances.pred_boxes.tensor.numpy()
        recs = instances.recs.numpy()

        results = []
        for bezier, rec, score in zip(beziers, recs, scores):
            if hasattr(instances, 'beziers'):
                # convert beziers to polygons
                poly = self.bezier_to_polygon(bezier)
            else:
                b = bezier.astype('float')
                poly = [[b[0], b[1]], [b[0], b[3]], [b[2],b[3]], [b[2], b[1]]]
            s = self.decode(rec)
            result = {
                "image_id": img_id,
                "category_id": 1,
                "polys": poly,
                "rec": s,
                "score": score,
            }
            results.append(result)
        return results


    def bezier_to_polygon(self, bezier):
        u = np.linspace(0, 1, 20)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = np.outer((1 - u) ** 3, bezier[:, 0]) \
            + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
            + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
            + np.outer(u ** 3, bezier[:, 3])
        
        # convert points to polygon
        points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
        return points.tolist()

    def ctc_decode(self, rec):
        # ctc decoding
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            elif c == self.voc_size -1:
                s += u'口'
            else:
                last_char = False
        return s
    
    
    def decode(self, rec):
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if self.voc_size ==96:
                    s += self.CTLABELS[c]
                else:
                    s += str(chr(self.CTLABELS[c]))
            elif c == self.voc_size -1:
                s += u'口'
    
        return s
