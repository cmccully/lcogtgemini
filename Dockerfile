FROM docker.lco.global/miniconda2:4.2.12
MAINTAINER Curtis McCully <cmccully@lco.global>

RUN yum -y install epel-release gcc glibc.i686\
        && yum -y clean all

RUN conda install -y pip numpy astropy ipython matplotlib scipy statsmodels \
        && conda install -y -c http://ssb.stsci.edu/astroconda iraf-all pyraf-all stsci gemini \
        && conda clean -y --all

RUN pip install astroscrappy \
        && rm -rf ~/.cache/pip

RUN mkdir /home/gemini && /usr/sbin/groupadd -g 10000 "domainusers" \
        && /usr/sbin/useradd -g 10000 -d /home/gemini -M -N gemini \
        && chown -R gemini:domainusers /home/gemini

WORKDIR /lco/

RUN git clone https://github.com/cmccully/pf_model.git

WORKDIR /lco/pf_model
RUN python /lco/pf_model/setup.py install

WORKDIR /lco/gemini

COPY . /lco/gemini
RUN python /lco/gemini/setup.py install

RUN yum -y install  xyeyes

USER gemini

RUN mkdir /home/gemini/iraf

ENV HOME=/home/gemini iraf=/opt/conda/iraf/ IRAFARCH=linux IRAF_EXTPKG=/opt/conda/extern.pkg TERM=xgterm

WORKDIR /home/gemini/iraf

RUN mkiraf -f

WORKDIR /home/gemini
