from PIL import Image
import os
import sys
import datetime
import logging

def compare_pixels(img1, img2, aug_name, width, height, image_offset = 0):
    pixel_difference = [0,0,0,0,0,0]
    if'rgb' in aug_name:
        pixels1 = img1.load()
        pixels2 = img2.load()
        total_valid_pixel_count = width * height * 3
        for wt in range(width):
            for ht in range(height):
                ht = ht + image_offset
                if pixels1[wt,ht] != pixels2[wt,ht]:
                    r = abs(pixels1[wt,ht][0]-pixels2[wt,ht][0])
                    g = abs(pixels1[wt,ht][1]-pixels2[wt,ht][1])
                    b = abs(pixels1[wt,ht][2]-pixels2[wt,ht][2])
                    if(r > 4):
                        r = 5
                    if(g > 4):
                        g = 5
                    if(b > 4):
                        b = 5
                    pixel_difference[r] += 1
                    pixel_difference[g] += 1
                    pixel_difference[b] += 1
                else:
                    pixel_difference[0] += 3
    else:
        pixels1 = img1.convert('L').load()
        pixels2 = img2.convert('L').load()
        total_valid_pixel_count = width * height * 1
        for wt in range(width):
            for ht in range(height):
                ht = ht + image_offset
                if pixels1[wt,ht] != pixels2[wt,ht]:
                    pixel = abs(pixels1[wt,ht]-pixels2[wt,ht])
                    if(pixel > 4):
                        pixel = 5
                    pixel_difference[pixel] += 1
                else:
                    pixel_difference[0] += 1
    return pixel_difference, total_valid_pixel_count

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    handlers = [logging.FileHandler('../log_file_' + timestamp + '.log'), logging.StreamHandler()]
    logging.basicConfig(level = logging.INFO, handlers = handlers)
    if( len(sys.argv) < 3 ):
        logging.error('Pass atleast 2 arguments (python image_comparison.py <ref_output_folder_path> <rocal_ouput_folder_path>')
        exit()

    # Open the two images
    ref_output_path = sys.argv[1]
    rocal_output_path = sys.argv[2]

    if not (os.path.exists(ref_output_path) and os.path.exists(rocal_output_path)):
        logging.error("Path does not Exists")
        exit()

    total_case_count = 0
    passed_case_count = 0
    failed_case_count = 0
    failed_case_list = []
    golden_output_dir_list = os.listdir(ref_output_path)
    rocal_output_dir_list = os.listdir(rocal_output_path)
    randomized_augmentation = ["Snow", "Rain", "Jitter", "SNPNoise"]
    golden_file_path = ""
    for aug_name in rocal_output_dir_list:
        temp = aug_name.split('.')
        file_name_s = temp[0].split('_')
        if(len(file_name_s) > 3):
            file_name_s.pop()
            golden_file_path = "_".join(file_name_s) + ".png"
        else:
            golden_file_path = aug_name

        #For randomized augmentation
        if(file_name_s[0] in randomized_augmentation):
            total_case_count=total_case_count+1
            augmentation_name = aug_name.split('.')[0]
            logging.info("Running %s",augmentation_name)
            passed_case_count=passed_case_count+1
            logging.info("PASSED ")
        elif golden_file_path in golden_output_dir_list:
            total_case_count=total_case_count+1
            ref_file_path = ref_output_path+golden_file_path
            rocal_file_path = rocal_output_path+aug_name
            if(os.path.exists(rocal_file_path) and os.path.exists(ref_file_path)):
                logging.info("Running %s ",aug_name.split('.')[0])
                img1 = Image.open(ref_file_path)
                img2 = Image.open(rocal_file_path)

                # Check if the images have the same dimensions
                if img1.size != img2.size:
                    logging.info("The images have different sizes.")
                    exit()

                # Compare the pixel values for each image
                pixeldiff = None
                tot_count = 0
                if 'larger' in aug_name:
                    resize_width = 400
                    resize_height = 300
                    image_offset = 400
                    pixeldiff, tot_count = compare_pixels(img1, img2, aug_name, resize_width, resize_height)
                    pixeldiff2, tot_count2 = compare_pixels(img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixeldiff = [x + y for x, y in zip(pixeldiff, pixeldiff2)]
                    tot_count = tot_count + tot_count2
                elif 'smaller' in aug_name:
                    resize_width = 533
                    resize_height = 400
                    image_offset = 2400
                    pixeldiff, tot_count = compare_pixels(img1, img2, aug_name, resize_width, resize_height)
                    pixeldiff2, tot_count2 = compare_pixels(img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixeldiff = [x + y for x, y in zip(pixeldiff, pixeldiff2)]
                    tot_count = tot_count + tot_count2
                else:
                    pixeldiff, tot_count = compare_pixels(img1, img2, aug_name, img1.size[0], img1.size[1])
                total_pixel_diff = 0
                for pix_diff in range(1,6):
                    total_pixel_diff += pixeldiff[pix_diff]
                mismatch_percentage = round((total_pixel_diff/tot_count)*100,2)
                if(total_pixel_diff == 0 or (mismatch_percentage < 5.0 and pixeldiff[1] == total_pixel_diff) or          # Ignore test cases with single pixel differences less than 5% of total pixel count
                   (mismatch_percentage < 0.5 and ('Blend' in aug_name or 'Rotate' in aug_name) and 'hip' in aug_name)): # Ignore mismatch in rotate augmentation less than 0.5% of total pixel count
                    passed_case_count=passed_case_count+1
                    logging.info("PASSED")
                else:
                    failed_case_list.append(golden_file_path)
                    failed_case_count=failed_case_count+1
                    logging.info("FAILED")
                    logging.info("Printing pixel mismatch %s",pixeldiff)
                    logging.info("Mismatach percentage %0.2f", mismatch_percentage)
                    for pix_diff in range(1,6):
                            logging.info("Percentage of %d pixel mismatch %0.2f", pix_diff, round((pixeldiff[pix_diff]/total_pixel_diff)*100,2))
            else:
                logging.info("Skipping the testcase as file not found %s",rocal_file_path)
        else:
            logging.info("File not found in ref_output_folder %s", golden_file_path)
    if len(failed_case_list) != 0:
        logging.info("Failing cases: {}".format(", ".join(failed_case_list)))
    logging.info("Total case passed --> {} / {} ".format(passed_case_count,total_case_count))
    logging.info("Total case failed --> {} / {} ".format(failed_case_count,total_case_count))

if __name__ == '__main__':
    main()
