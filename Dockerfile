FROM docker.lco.global/miniconda2:4.2.12
MAINTAINER Curtis McCully <cmccully@lco.global>

RUN yum -y install epel-release gcc glibc.i686 xorg-x11-apps mesa-libGL.x86_64 mesa-libGL.i686 \
        && yum -y clean all

# Fix for x11 forwarding
RUN dbus-uuidgen > /etc/machine-id

RUN mkdir /home/gemini && /usr/sbin/groupadd -g 10000 "domainusers" \
        && /usr/sbin/useradd -g 10000 -d /home/gemini -M -N -u 10197 gemini \
        && chown -R gemini:domainusers /home/gemini && chown -R gemini:domainusers /opt/conda

USER gemini

RUN conda install -y pip numpy astropy ipython matplotlib scipy statsmodels \
        && conda install -y -c http://ssb.stsci.edu/astroconda iraf-all pyraf-all stsci gemini \
        && conda clean -y --all

RUN pip install astroscrappy \
        && rm -rf ~/.cache/pip

RUN mkdir /home/gemini/bin \
        && mkdir /home/gemini/src

RUN wget http://www.gemini.edu/sciops/data/software/gmoss_fix_headers.py -O /home/gemini/bin/gmoss_fix_headers.py \
        && chmod +x /home/gemini/bin/gmoss_fix_headers.py

RUN git clone https://github.com/cmccully/pf_model.git /home/gemini/src/pf_model

WORKDIR /home/gemini/src/pf_model

RUN python /home/gemini/src/pf_model/setup.py install

USER root
COPY . /home/gemini/src/gemini
RUN chown -R gemini:domainusers /home/gemini/src/gemini
USER gemini

WORKDIR /home/gemini/src/gemini

RUN python /home/gemini/src/gemini/setup.py install

RUN mkdir /home/gemini/iraf

ENV HOME=/home/gemini iraf=/opt/conda/iraf/ IRAFARCH=linux IRAF_EXTPKG=/opt/conda/extern.pkg TERM=xgterm \
        PATH="/home/gemini/bin/:${PATH}"

WORKDIR /home/gemini/iraf

RUN mkiraf -f

WORKDIR /home/gemini
