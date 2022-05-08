import base64

import exifread
from gmssl import func

file_path = "C:/Users/Administrator/Desktop/IMAG0001.jpg"
def get_image_exif(filepath):
    f = open(filepath, 'rb')
    tags = exifread.process_file(f)
    info = str(tags.get('EXIF UserComment', '0').values)

    info = bytes.decode('utf-8')
    print(info)
    print(1)
    #info = tags.get('EXIF UserComment', '0').values

    # info = {
    #     'EXIF UserComment':tags.get('EXIF UserComment', '0').values
    # }
    f.close()
   # print(info)
    return info

def Analysis_UserComment(strUserCom):
    #strtemp = "CANGLU;CL_A1;SD37;SDNR010X;KYS004;2022/01/08 14:11:17;25.3\xa1\xe3C;48.9%;Waxing Crescent Moon;E 127.29.30 N 25.81.01;938m;Bare Rock;Secondary Forest;Shrub;None;None;66%;'"
    strtemp = strUserCom

    strtemp = strtemp.split(';')
    print(strtemp)
    return strtemp

ss = get_image_exif(file_path)
#Analysis_UserComment(ss)