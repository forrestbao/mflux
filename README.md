LICENSE
========
GPL V3. 
Copyleft 2014 to 2022 Forrest Sheng Bao et al. 


SETUP
=====

Assume root directory is `/var/www/mflux`.

Python environment
------------------
1. `sudo apt-get install python-dev`
<!-- 2. `sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose` -->
3. `python3 -m pip install scikit-learn python-constraint sympy`
4. regenerate all models by running `python get_model.py`


# Web server 

```shell
sudo apt install apache2
sudo a2enmod cgi
```

Besure that `cgi.conf` and `cgi.load` are under `/etc/apache2/mods-enabled`

Add the following content to `/etc/apache2/apache2.conf`: 
```
<Directory "/var/www/mflux">
AllowOverride None
Options +ExecCGI 
Require all granted
Allow from all
AddHandler cgi-script .py              # tell Apache to handle every file with .py suffix as a cgi program
AddHandler default-handler .html .htm  # tell Apache to handle HTML files in regular way
</Directory>
```

Be sure that `DocumentRoot` is set under `/etc/apache2/sites-enabled`, like 
```
        DocumentRoot /var/www/mflux
```

And restart the web server: 
```shell
sudo systemctl restart apache2
```
