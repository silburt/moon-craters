import numpy as np
from skimage.feature import match_template
import cv2

###########HYPERPARAMETERS###########
# LONGLAT_THRESH2, RAD_THRESH: for template matching, if (x1-x2)^2 + (y1-y2)^2 < longlat_thresh2 AND abs(r1-r2) < max(1.01,rad_thresh*r1), remove (x2,y2,r2) circle (it is a duplicate of another crater candidate). In addition, during predicted target -> csv matching (i.e. template_match_target_to_csv), the same criteria is used to match CNN craters with csv craters (increasing the recall). Maybe these should technically be separate parameters, but to first order they should be the same...
# TEMPLATE_THRESH: 0-1 range, if scikit-image's template matching probability > template_thresh, count as detection
# TARGET_THRESH: 0-1 range, set pixel values > target_thresh to 1, and pixel values < target_thresh -> 0
# MINRAD/MAXRAD are the radii to search over during template matching. For minrad, keep in mind that if the predicted target has thick rings, a small ring of diameter ~ rw could be detected by match_filter.
minrad_ = 3
maxrad_ = 50
longlat_thresh2_ = 70
rad_thresh_ = 0.3
template_thresh_ = 0.5
target_thresh_ = 0.1

#####################################
def template_match_t(target, minrad=minrad_, maxrad=maxrad_,
                     longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_,
                     template_thresh=template_thresh_,
                     target_thresh=target_thresh_):
    """Extracts crater coordinates (in pixels) from a CNN-predicted target by
    iteratively sliding rings through the image via match_template from
    scikit-image.

    Parameters
    ----------
    target : array
        CNN-predicted target.
    minrad : integer
        Minimum ring radius to search target over.
    maxrad : integer
        Maximum ring radius to search target over.
    longlat_thresh2 : float
        Minimum squared longitude/latitude difference between craters to be
        considered distinct detections.
    rad_thresh : float
        Minimum fractional radius difference between craters to be considered
        distinct detections.
    template_thresh : float
        Minimum match_template correlation coefficient to count as a detected
        crater.
    target_thresh : float
        Value between 0-1. All pixels > target_thresh are set to 1, and
        otherwise set to 0.

    Returns
    -------
    coords : array
        Pixel coordinates of successfully detected craters in predicted target.
    """

    # thickness of rings for the templates.
    rw = 2

    # threshold target
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0

    radii = np.arange(minrad, maxrad + 1, 1, dtype=int)
    coords = []     # coordinates extracted from template matching
    corr = []       # correlation coefficient for coordinates set
    for r in radii:
        # template
        n = 2 * (r + rw + 1)
        template = np.zeros((n, n))
        cv2.circle(template, (r + rw + 1, r + rw + 1), r, 1, rw)

        # template match - result is nxn array of probabilities
        result = match_template(target, template, pad_input=True)
        index_r = np.where(result > template_thresh)
        coords_r = np.asarray(zip(*index_r))
        corr_r = np.asarray(result[index_r])

        # store x,y,r
        try:
            for c in coords_r:
                coords.append([c[1], c[0], r])
            for l in corr_r:
                corr.append(np.abs(l))
        except:
            pass

    # remove duplicates from template matching at neighboring radii/locations
    coords, corr = np.asarray(coords), np.asarray(corr)
    i, N = 0, len(coords)
    while i < N:
        Long, Lat, Rad = coords.T
        lo, la, r = coords[i]
        diff_longlat = (Long - lo)**2 + (Lat - la)**2
        diff_rad = abs(Rad - r)
        index = (diff_rad < max(2.01, rad_thresh * r)) & (diff_longlat < longlat_thresh2)
        if len(np.where(index == True)[0]) > 1:
            # replace current coord with max match probability coord in
            # duplicate list
            coords_i = coords[np.where(index == True)]
            corr_i = corr[np.where(index == True)]
            coords[i] = coords_i[corr_i == np.max(corr_i)][0]
            index[i] = False
            coords = coords[np.where(index == False)]
        N, i = len(coords), i + 1

    return coords


def template_match_t2c(target, csv_coords, minrad=minrad_, maxrad=maxrad_,
                       longlat_thresh2=longlat_thresh2_,
                       rad_thresh=rad_thresh_, template_thresh=template_thresh_,
                       target_thresh=target_thresh_, rmv_oob_csvs=0):
    """Extracts crater coordinates (in pixels) from a CNN-predicted target and
    compares the resulting detections to the corresponding human-counted crater
    data.

    Parameters
    ----------
    target : array
        CNN-predicted target.
    csv_coords : array
        Human-counted crater coordinates (in pixel units).
    minrad : integer
        Minimum ring radius to search target over.
    maxrad : integer
        Maximum ring radius to search target over.
    longlat_thresh2 : float
        Minimum squared longitude/latitude difference between craters to be
        considered distinct detections.
    rad_thresh : float
        Minimum fractional radius difference between craters to be considered
        distinct detections.
    template_thresh : float
        Minimum match_template correlation coefficient to count as a detected
        crater.
    target_thresh : float
        Value between 0-1. All pixels > target_thresh are set to 1, and
        otherwise set to 0.
    rmv_oob_csvs : boolean, flag
        If set to 1, remove craters from the csv that are outside your
        detectable range.

    Returns
    -------
    N_match : int
        Number of crater matches between your target and csv.
    N_csv : int
        Number of csv entries
    N_detect : int
        Total number of detected craters from target.
    maxr : int
        Radius of largest crater extracted from target.
    err_lo : float
        Mean longitude error between detected craters and csvs.
    err_la : float
        Mean latitude error between detected craters and csvs.
    err_r : float
        Mean radius error between detected craters and csvs.
    csv_duplicates : list
        List of multiple csv entries that matched to single detected crater.
    """
    # get coordinates from template matching
    templ_coords = template_match_t(target, minrad, maxrad, longlat_thresh2,
                                    rad_thresh, template_thresh, target_thresh)

    # find max detected crater radius
    maxr = 0
    if len(templ_coords > 0):
        maxr = np.max(templ_coords.T[2])

    # compare template-matched results to ground truth csv input data
    N_match = 0
    csv_duplicates = []
    err_lo, err_la, err_r = 0, 0, 0
    N_csv, N_detect = len(csv_coords), len(templ_coords)
    for lo, la, r in templ_coords:
        csvLong, csvLat, csvRad = csv_coords.T
        diff_longlat = (csvLong - lo)**2 + (csvLat - la)**2
        diff_rad = abs(csvRad - r)
        index = (diff_rad < max(1.01, rad_thresh * r)) & (diff_longlat < longlat_thresh2)
        index_True = np.where(index == True)[0]
        N = len(index_True)
        if N > 1:
            cratervals = np.array((lo, la, r))
            id_keep = index_True[0]
            diff = np.sum((csv_coords[id_keep] - cratervals)**2)
            csv_duplicates.append(csv_coords[id_keep])
            for id in index_True[1:]:
                dupevals = csv_coords[id]
                index[id] = False
                csv_duplicates.append(dupevals)
                diff_ = np.sum((dupevals - cratervals)**2)
                if diff_ < diff:
                    id_keep = id
                    diff = diff_
            index[id_keep] = True       # keep only closest match as true
            Lo, La, R = csv_coords[id_keep].T
            err_lo += abs(Lo - lo) / r
            err_la += abs(La - la) / r
            err_r += abs(R - r) / r
            print("%d GT entries matched to (%d,%d,%d) ring... counted " /
                  "(%f,%f,%f) as the match." % (N, lo, la, r, Lo, La, r))
            print(csv_duplicates)
        elif N == 1:
            Lo, La, R = csv_coords[index_True[0]].T
            err_lo += abs(Lo - lo) / r
            err_la += abs(La - la) / r
            err_r += abs(R - r) / r
        N_match += min(1, N)
        # remove csv so it can't be re-matched again
        csv_coords = csv_coords[np.where(index == False)]
        if len(csv_coords) == 0:
            break

    if rmv_oob_csvs == 1:
        N_large_unmatched = len(np.where((csv_coords.T[2] > maxr) |
                                         (csv_coords.T[2] < minrad_))[0])
        if N_large_unmatched < N_csv:
            N_csv -= N_large_unmatched

    if N_match >= 1:
        err_lo = err_lo / N_match
        err_la = err_la / N_match
        err_r = err_r / N_match

    return N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, csv_duplicates
