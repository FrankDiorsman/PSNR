from multiprocessing import Pool, cpu_count
from skimage.feature import greycomatrix, greycoprops
from skimage.util.shape import view_as_windows

def warp_greycomatrix(arr, dist=None, angle=None, level=None):
    return greycomatrix(arr, [dist], [angle], levels=level)    # , np.pi/4, np.pi/2, np.pi*3/4

def warp_greycoprops(arr, feature_type=None, level=None):
    if arr.size < level*level: return np.array([[0]], dtype=np.float)
    arr = arr.reshape(level,level,1,1)
    return greycoprops(arr, feature_type)

def sub_process3(infile, window_size, distance, angle, level, feature_type, outfile):

    try:
        in_ds = gdal.Open(infile)
        ns, nl, nb = in_ds.RasterXSize, in_ds.RasterYSize, in_ds.RasterCount
        in_band = in_ds.GetRasterBand(1)
        arr = in_band.ReadAsArray()

        # 依据移动窗口大小扩展输入数组
        pad = window_size // 2
        new_arr = np.pad(arr, ((pad, pad), (pad, pad)), "edge")

        # 依据扩展数组获取各个移动窗口
        kernel_arrs = view_as_windows(new_arr, (window_size, window_size), step=1)
        del arr, new_arr, in_band, in_ds

        nl, ns, _, _ = kernel_arrs.shape
        feat_arr = np.zeros([nl, ns], dtype=np.float)

        print(angle, "enter...")
        for j in range(0, nl): # ,500
            # if j + 500 < nl:
            #     rows = 500
            # else:
            #     rows = nl - j
            # temp_arr = np.concatenate(np.concatenate(kernel_arrs[j:(j+rows), :], axis=1), axis=1)
            temp_arr = np.concatenate(kernel_arrs[0,:], axis=1)
            da_arr = da.from_array(temp_arr, chunks=(window_size, window_size))
            da_matrix = da_arr.map_overlap(warp_greycomatrix, 0, boundary="nearest", dist=distance, angle=angle,
                                           level=level)
            glcms = da_matrix.compute()
            print(j, "Finish GLCM")
            da_arr = da.from_array(glcms, chunks=(level, level))
            da_props = da_arr.map_overlap(warp_greycoprops, 0, boundary="nearest", feature_type=feature_type, level=level)
            props = da_props.compute()
            print(j, "Finish Texture")
            feat_arr[j, :] += props.ravel()
            del da_arr, da_matrix, glcms, da_props, props
        print(angle, "end")

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(outfile, ns, nl, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(feat_arr)
        del feat_arr, out_band, out_ds

    except Exception as error_msg:
        print(error_msg)