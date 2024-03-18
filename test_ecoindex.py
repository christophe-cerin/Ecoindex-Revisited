#!/usr/bin/env python3
"""Example file to illustrate the eco_index Computation through
the historical method, as explained in
 https://github.com/cnumr/GreenIT-Analysis

Extra works:
   - take into consideration the sizes of downloaded fonts,
     .css and .js files

$ python3 test_eco_index.py http://www.google.com
and the reply is
http://www.google.com ; 80 ; 12 ; 19254 ; 90.97 ; 1.18 ; 1.77
with the URL, the DOM, requests, size, eco_index, Water, Gas emission
"""

from html.parser import HTMLParser
from re import search
from re import finditer
import re
import requests
from requests.exceptions import HTTPError
import pyparsing
from htmldom import htmldom
import sys
import AdvancedHTMLParser

# CSS parser documentation on https://tinycss.readthedocs.io/en/latest/css3.html
import tinycss

# Configuration file
import configparser



__author__ = "Christophe Cerin"
__copyright__ = "Copyright 2022"
__credits__ = ["Christophe Cerin"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Christophe Cerin"
__email__ = "christophe.cerin@univ-paris13.fr"
__status__ = "Experimental"

# For an old version: URL used for the eco_index computation
#myurl = 'https://www.lipn.univ-paris13.fr/~cerin/'
#myurl = 'http://www.google.com'
#myurl = 'http://datamove.imag.fr/denis.trystram/'
#myurl = 'https://www.cgi.com/'
#myurl = 'https://www.smile.eu/'

# To solve the 403 http request error add the lines, but with no
# guaranty on the result
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}
headers = {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    }

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Read the verbose level
verbose_level = config.getint('Verbose', 'level')

# Print the verbose level
if verbose_level > 0:
    print('Verbose level: ', verbose_level)

# Cache of http requests. We assume that, if a http request is in the cache, it is
# not necessary to fetch it again (it is in the browser cache)
cache_url = {}

# Total sum of the bytes read
nb_bytes_read = 0

# Number of http requests
nb_http_requests = 0

""" A function to compute the number of bytes that each http(s) request
    do in a css file (for instance to download fonts)
"""
def explore_css(myurl):
    global nb_http_requests
    result = 0
    for url in [myurl]:
        try:
            response = requests.get(url,headers=headers)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            if verbose_level > 1:
                print(f'HTTP error occurred: {http_err}')  # Python 3.6
            pass
        except Exception as err:
            if verbose_level > 1:
                print(f'Other error occurred: {err}')  # Python 3.6
            pass
        else:
            if verbose_level>1:
                print('Success for: ',url)
            if url not in cache_url:
                cache_url[url] = True
                result += len(response.text)
                nb_http_requests += 1
                if verbose_level>1:print('Nb bytes read: ',len(response.text))
                # We cancel the comments in the css file
                comment = pyparsing.nestedExpr("/*", "*/").suppress()
                if verbose_level>1:print('Css file WITHOUT comment',comment.transformString(response.text))
                # No: pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',re.MULTILINE|re.DOTALL)
                #pattern = re.compile(r'(https|ftp|http|ftps):\/\/([\~A-Za-z\d_]+\.)?(([\~a-zA-Z\d_]+)(\.[\~a-zA-Z]{2,6}))(\/[\~a-zA-Z\d_\%\-=\+\.]+)*(\?)?([a-zA-Z\~\d=_\+\%\-&\{\}\:\.]+)?',re.MULTILINE|re.DOTALL)
                #pattern = re.compile(r'(https|ftp|http|ftps):\/\/([\~A-Za-z\d_]+\.)?(([\~a-zA-Z\d_]+)(\.[\~a-zA-Z]{2,6}))(\/[\~a-zA-Z\d_\%\-=\+\.]+)*(\?)?([a-zA-Z\~\d=_\+\%\-&\{\}\:\.]+)?',re.MULTILINE|re.DOTALL)
                pattern = re.compile(r'http[\d\-:\.\~a-zA-Z\/=_\+\%\-=_\+\%\-&\{\}\?\:]+',re.MULTILINE|re.DOTALL)
                matches = finditer(pattern,comment.transformString(response.text))
                for m in matches:
                    if verbose_level>1:print('In css: ',m.group())
                    if m.group() not in cache_url:
                        cache_url[m.group()] = True
                        # Check if font
                        if(search("^.*\.svg$",m.group()) or
                            search("^.*\.woff$",m.group()) or
                            search("^.*\.eot$",m.group())  or
                            search("^.*\.ttf$",m.group())):
                            
                            result += explore_css(m.group())
                    #else:
                         #pass
                         #print("Url still fetched!")
    return result

""" A class to parse the HTML file and to count the number of http requests
"""
class Parser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.in_p = []
        self.in_a = []
    """ A function to handle the start tag of the HTML file
    """
    def handle_starttag(self, tag, attrs):
        if verbose_level>1: print("Start tag:", tag)
        global nb_bytes_read
        global nb_http_requests
        for attr in attrs:
            if verbose_level>1:print("     attr:", attr)
            if tag == 'a' and attr[0] == 'href' and search("^http",attr[1]):
                if verbose_level>1:print("    href tag with http request on: ",attr[1])
                nb_http_requests += 1
            if tag == 'link' and attr[0] == 'href' and search("^http",attr[1]):
                if verbose_level>1:print("    link tag with http request on: ",attr[1])
                if search("^.*\.css$",attr[1]):
                    if verbose_level>1:print("    Css file")
                    nb_bytes_read += explore_css(attr[1])
                else:
                    for url in [attr[1]]:
                        try:
                            response_css = requests.get(url,headers=headers)
                            # If the response was successful, no Exception will be raised
                            response_css.raise_for_status()
                        except HTTPError as http_err:
                            if verbose_level>1:
                                print(f'HTTP error occurred: {http_err}')  # Python 3.6
                            pass
                        except Exception as err:
                            if verbose_level>1:
                                print(f'Other error occurred: {err}')  # Python 3.6
                            pass
                        else:
                            if url not in cache_url:
                                if verbose_level>1:print("    Not a css file")
                                nb_bytes_read += len(response_css.text)
                                nb_http_requests += 1
                if verbose_level>1:print('Nb bytes read: ',nb_bytes_read)
            if tag == 'img' and attr[0] == 'src':
                if (search("http:",attr[1]) or
                     search("https:",attr[1])):
                    foo = attr[1]
                else:
                    foo = str(myurl+attr[1])
                if verbose_level>1:print("    src tag with http request on: ",foo, "attr[1]: ",attr[1], " ***")
                for url in [foo]:
                    try:
                        response_img = requests.get(url,headers=headers)
                        # Test: response_img = requests.get('https://www.w3schools.com/images/picture.jpg')
                        # If the response was successful, no Exception will be raised
                        response_img.raise_for_status()
                    except HTTPError as http_err:
                        if verbose_level>1:
                            print(f'HTTP error occurred: {http_err}')  # Python 3.6
                        pass
                    except Exception as err:
                        if verbose_level>1:
                            print(f'Other error occurred: {err}')  # Python 3.6
                        pass
                    else:
                        if url not in cache_url:
                            nb_bytes_read += len(response_img.text)
                            nb_http_requests += 1
                if verbose_level>1:print('Nb bytes read: ',nb_bytes_read)
            if tag == 'script' and attr[0] == 'src' and search("^http",attr[1]):
                if verbose_level>1:print("    script tag with http request on: ",attr[1])
                if search("^.*\.js$",attr[1]):
                    if verbose_level>1:print("    Js file")
                    nb_bytes_read += explore_css(attr[1])
                else:
                    for url in [attr[1]]:
                        try:
                            response_js = requests.get(url,headers=headers)
                            # If the response was successful, no Exception will be raised
                            response_js.raise_for_status()
                        except HTTPError as http_err:
                            if verbose_level>1:
                                print(f'HTTP error occurred: {http_err}')  # Python 3.6
                            pass
                        except Exception as err:
                            if verbose_level>1:
                                print(f'Other error occurred: {err}')  # Python 3.6
                            pass
                        else:
                            if url not in cache_url:
                                if verbose_level>1: print("    Not a js file")
                                nb_bytes_read += len(response_js.text)
                                nb_http_requests += 1
                if verbose_level>1:print('Nb bytes read: ',nb_bytes_read)
        if (tag == 'p'):
            self.in_p.append(tag)
        if (tag == 'a'):
            self.in_a.append(tag)

    #def handle_endtag(self, tag):
    #            if (tag == 'p'):
    #                    self.in_p.pop()
    #            if (tag == 'a'):
    #                    self.in_a.pop()

    #def handle_data(self, data):
    #            if self.in_p:
    #                    print("<p> data :", data)
    #            if self.in_a:
    #                    print("<a> data :", data)


#
# Calcul eco_index based on formula from web site www.eco_index.fr
#

quantiles_dom = [
    0, 47, 75, 159, 233, 298, 358,
    417, 476, 537, 603, 674, 753,
    843, 949, 1076, 1237, 1459, 1801, 2479, 594601,
    ]
quantiles_req = [
    0, 2, 15, 25, 34, 42, 49,
    56, 63, 70, 78, 86, 95,
    105, 117, 130, 147, 170, 205, 281, 3920,
    ]
quantiles_size = [
    0, 1.37, 144.7, 319.53, 479.46, 631.97, 783.38,
    937.91, 1098.62, 1265.47, 1448.32, 1648.27, 1876.08,
    2142.06, 2465.37, 2866.31, 3401.59, 4155.73, 5400.08,
    8037.54, 223212.26,
    ]


""" A function to compute the eco_index
"""
def compute_eco_index(dom,req,size):
    q_dom = compute_quantile(quantiles_dom,dom)
    q_req = compute_quantile(quantiles_req,req)
    q_size= compute_quantile(quantiles_size,size)
    return 100 - 5 * (3*q_dom + 2*q_req + q_size)/6


""" A function to compute the quantile
"""
def compute_quantile(quantiles,value):
    for i in range(1,len(quantiles)):
        if value < quantiles[i]:
            return (i -1 + (value-quantiles[i-1])/(quantiles[i] -quantiles[i-1]))
    return len(quantiles) - 1


""" A function to compute the eco_index Grade
"""
def get_eco_index_grade(eco_index):
    if (eco_index > 80):
        return "A"
    elif (eco_index > 70):
        return "B"
    elif (eco_index > 55):
        return "C"
    elif (eco_index > 40):
        return "D"
    elif (eco_index > 25):
        return "E"
    elif (eco_index > 10):
        return "F"
    return "G"


""" A function to compute the Greenhouse Gases Emission from eco_index
"""
def compute_greenhouse_gases_emission_from_eco_index(eco_index):
    return '{:.2f}'.format(2 + 2 * (50 - eco_index) / 100)


"""" A function to compute the Water Consumption from eco_index
"""
def compute_water_consumption_from_eco_index(eco_index):
    return '{:.2f}'.format(3 + 3 * (50 - eco_index) / 100)


#
# Main
#
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Bad number of argument. Require an URL as parameter!")
        print('usage: python3 eco_index.py <url>')
        exit()

    if verbose_level>1: print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if verbose_level>1:print(f"Argument {i:>6}: {arg}")
        if i == 1:
            myurl = arg

    # First, we download the html for the myurl URL
    for url in [myurl]:
        if url not in cache_url:
            try:
                response = requests.get(url,headers=headers)
                # If the response was successful, no Exception will be raised
                response.raise_for_status()
            except HTTPError as http_err:
                if verbose_level>1:
                    print(f'HTTP error occurred: {http_err}')  # Python 3.6
                pass
            except Exception as err:
                if verbose_level>1:
                    print(f'Other error occurred: {err}')  # Python 3.6
                pass
            else:
                if verbose_level>1:print('Success!')
                nb_bytes_read = nb_bytes_read + len(response.text)
                cache_url[url] = True

                # Then we explore the HTML file, token by token
                # We specialize on href, link and img/src tokens
                parser = Parser()
                parser.feed(response.text)
                if verbose_level>1:print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
                if verbose_level>1:print('Total sum of the bytes read: ',nb_bytes_read)

                # DOM exploration: HtmlDom seems buggy!
                # Observation made by C. Cerin on Oct 31, 2022 
                # We do prefer to use AdvancedHTMLParser library!
                dom = htmldom.HtmlDom("http://www.google.com").createDom()
                try:
                    dom = htmldom.HtmlDom(myurl).createDom()
                    parser = AdvancedHTMLParser.AdvancedHTMLParser()
                    parser.parseStr(response.text)
                    items = parser.getAllNodes()
                    res = len(items)
                    pass
                except:
                    sys.exit()

                # Find all the links present on a page and prints its "href" value
                if verbose_level>1:
                    a = dom.find( "a" )
                    for link in a:
                            print( link.attr( "href" ) )

                # Getting all elements
                    all = dom.find("*")
                    #Compute number of elements in dom
                    res = 0
                    for i in all:
                        print(i.len)
                    res += i.len
                    print('Number of elements in DOM: ',res)

                    print('Number of http requests: ',nb_http_requests)

                    print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

                    print('URL:                                      ',myurl)

                # Nota: the third parameter should be in kB, hence the division by 1024
                eco_index = compute_eco_index(res, nb_http_requests, int(nb_bytes_read/1024))
                if verbose_level>1:
                    print('eco_index:                                 ','{:.2f}'.format(eco_index))

                    print('eco_index Grade:                           ', get_eco_index_grade(eco_index))

                    print('Greenhouse Gases Emission from eco_index:  ',compute_greenhouse_gases_emission_from_eco_index(eco_index), ' (gCO2e)')

                    print('Water Consumption from eco_index:          ', compute_water_consumption_from_eco_index(eco_index), ' (cl)')

                    print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
                print(myurl,';',res,';',nb_http_requests,';',nb_bytes_read,';','{:.2f}'.format(eco_index),';',compute_greenhouse_gases_emission_from_eco_index(eco_index),';',compute_water_consumption_from_eco_index(eco_index))
