echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf d3graph.egg-info
rm -rf d3graph/__pycache__
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf .pylint.d
rm *.js
rm *.css
rm *.html

rm d3graph/*.js
rm d3graph/*.css
rm d3graph/*.html
