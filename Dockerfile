FROM docker.lco.global/miniconda2:4.2.12
MAINTAINER Curtis McCully <cmccully@lco.global>

RUN yum -y install epel-release gcc glibc.i686 xorg-x11-apps\
        && yum -y clean all

RUN conda install -y pip numpy astropy ipython matplotlib scipy statsmodels \
        && conda install -y -c http://ssb.stsci.edu/astroconda iraf-all pyraf-all stsci gemini \
        && conda clean -y --all

RUN pip install astroscrappy \
        && rm -rf ~/.cache/pip

RUN mkdir /home/gemini && /usr/sbin/groupadd -g 10000 "domainusers" \
        && /usr/sbin/useradd -g 10000 -d /home/gemini -M -N -u 10197 gemini \
        && chown -R gemini:domainusers /home/gemini

WORKDIR /lco/

RUN wget http://www.gemini.edu/sciops/data/software/gmoss_fix_headers.py \
        && chmod +x gmoss_fix_headers.py

RUN git clone https://github.com/cmccully/pf_model.git

WORKDIR /lco/pf_model
RUN python /lco/pf_model/setup.py install

WORKDIR /lco/gemini

COPY . /lco/gemini
RUN python /lco/gemini/setup.py install

USER gemini

RUN mkdir /home/gemini/iraf

ENV HOME=/home/gemini iraf=/opt/conda/iraf/ IRAFARCH=linux IRAF_EXTPKG=/opt/conda/extern.pkg TERM=xgterm \
        PATH="/lco/:${PATH}"

WORKDIR /home/gemini/iraf

RUN mkiraf -f

WORKDIR /home/gemini
