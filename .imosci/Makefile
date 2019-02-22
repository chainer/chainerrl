build: build-py3.stable-chainer build-py3.old-chainer \
       build-py2.stable-chainer build-py2.old-chainer \
	   build-py3.gpu
.PHONY: build

install: install-py3.stable-chainer install-py3.old-chainer \
         install-py2.stable-chainer install-py2.old-chainer \
		 install-py3.gpu
.PHONY: install

build-%:
	docker build -t imosci/chainerrl:$* -f $*.Dockerfile .
.PHONY: build

install-%:
	docker push imosci/chainerrl:$*
.PHONY: install
