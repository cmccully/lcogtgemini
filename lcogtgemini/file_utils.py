def getobstypes(fs):
    # get the type of observation for each file
    obstypes = []
    obsclasses = []
    for f in fs:
        obstypes.append(fits.getval(f, 'OBSTYPE', ext=0))
        obsclasses.append(fits.getval(f, 'OBSCLASS', ext=0))

    obstypes = np.array(obstypes)
    obsclasses = np.array(obsclasses)
    return obstypes, obsclasses


def getobjname(fs, obstypes):
    objname = fits.getval(fs[obstypes == 'OBJECT'][0], 'OBJECT', ext=0).lower()

    # get rid of nonsense in the name (like the plus and whitespace
    objname = objname.replace('+', '')
    objname = ''.join(objname.split())

    # replace ltt with just l
    objname = objname.replace('ltt', 'l')
    return objname


def getredorblue(f):
    return f.split('.')[1][1]


def getsetupname(f):
    # Get the setup base name by removing the exposure number
    return f.split('.')[0] + '.' + f.split('.')[1][1:]


def gettxtfiles(fs, objname):

    flatfiles = np.array(glob('*.flat.txt'))

    # reduce the CuAr arcfiles.  Not flat fielded, gaps are not fixpixed
    arcfiles = np.array(glob('*.arc.txt'))

    # reduce the science files
    scifiles = glob(objname + '*.txt')

    nonscifiles = []
    # remove the arcs and flats
    for f in scifiles:
        if 'arc' in f or 'flat' in f: nonscifiles.append(f)

    for f in nonscifiles:
        scifiles.remove(f)
    scifiles = np.array(scifiles)

    return flatfiles, arcfiles, scifiles


def maketxtfiles(fs, obstypes, obsclasses, objname):
    # go through each of the files (Ignore bias and aquisition files)
    goodfiles = np.logical_and(obsclasses != 'acqCal', obsclasses != 'acq')
    goodfiles = np.logical_and(goodfiles, obstypes != 'BIAS')

    for f in fs[goodfiles]:
        # put the filename in the correct text file.
        obsstr = ''
        obstype = fits.getval(f, 'OBSTYPE', ext=0)
        if obstype != 'OBJECT':
            obsstr = '.' + obstype.lower()
            expnum = ''
        else:
            expnum = 1

        # Drop the raw/
        fname = f.split('/')[-1]
        # red or blue setting
        redblue = fits.getval(f, 'GRATING')[0].lower()
        # central wavelength
        lamcentral = fits.getval(f, 'CENTWAVE')

        txtname = '%s.%s%s%i%s.txt' % (objname, str(expnum), redblue, lamcentral, obsstr)
        # If more than one arc or flat, append to the text file
        if os.path.exists(txtname):
            if obsstr == '.flat' or obsstr == '.arc':
                # write to a text file
                txtfile = open(txtname, 'a')
            else:
                # We need to increment the exposure number
                moreimages = True
                expnum += 1
                while moreimages:
                    txtname = '%s.%s%s%i%s.txt' % (objname, str(expnum), redblue, lamcentral, obsstr)
                    if not os.path.exists(txtname):
                        txtfile = open(txtname, 'w')
                        moreimages = False
                    else:
                        expnum += 1
        else:
            txtfile = open(txtname, 'w')

        txtfile.write(fname + '\n')
        txtfile.close()