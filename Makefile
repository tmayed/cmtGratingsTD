#############################################################

.PHONY: default push git_rm_cached clean install install_dev

#############################################################

APP = cmt_gratings

#############################################################

default:
	@echo "must specify a command"

#############################################################

push:
	git add .
	git commit -m "$(m)"
	git push -u origin $(b)

git_rm_cached:
	git rm --cached `git ls-files -i -c --exclude-from=.gitignore`

#############################################################

clean:
ifeq ($(OS),Windows_NT)
	cmd /c "IF EXIST .venv rmdir /s /q .venv && IF EXIST poetry.lock del /f /q poetry.lock"
else
	rm -rf poetry.lock
endif
	
install:
	poetry install

install_dev:
	poetry install --with dev

reinstall:
	$(MAKE) clean
	$(MAKE) install

reinstall_dev:
	$(MAKE) clean
	$(MAKE) install_dev

#############################################################

pylint:
	poetry run pylint $(APP)

black:
	poetry run black $(APP)

#############################################################

bragg:
	poetry run python $(APP)/cmt_bragg_td_gauss_apod.py

#############################################################

moire:
	poetry run python $(APP)/cmt_moire_td_gauss_apod.py

#############################################################