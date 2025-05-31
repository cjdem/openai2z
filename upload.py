"""
Geographic data analysis and satellite image processing module
For analyzing TerraBrasilis data, geoglyphs, and satellite imagery
"""

import logging
import random
import pathlib
import os
import base64
import time
import json
import re
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
import geemap
from shapely.geometry import Point
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
# from zhipuai import ZhipuAI
from openai import OpenAI


class GeoDataProcessor:
    """Main class for processing geographic data"""

    def __init__(self, seed: int = 42, run_id: int = 1):
        """Initialize the processor"""
        self.seed = seed
        self.run_id = run_id
        self._setup_environment()
        self._setup_logging()

    def _setup_environment(self) -> None:
        """Set up environment variables and random seed"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        pathlib.Path("outputs").mkdir(exist_ok=True)
        load_dotenv()

    def _setup_logging(self) -> None:
        """Configure logging"""
        logging.basicConfig(
            filename="checkpoint2_run.log",
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    def load_terrabrasilis_data(self, file_path: str) -> gpd.GeoDataFrame:
        """Load TerraBrasilis KML data"""
        try:
            return gpd.read_file(file_path, driver='KML')
        except Exception as e:
            logging.error(f"Failed to load TerraBrasilis data: {e}")
            raise


class KMLParser:
    """KML file parser"""

    KML_NAMESPACE = "http://www.opengis.net/kml/2.2"

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.root = self._parse_kml()

    def _parse_kml(self) -> ET.Element:
        """Parse KML file"""
        try:
            with open(self.file_path, 'rt', encoding='utf-8') as f:
                doc = f.read()
            return ET.fromstring(doc)
        except Exception as e:
            logging.error(f"Failed to parse KML file: {e}")
            raise

    def extract_geoglyphs(self) -> List[ET.Element]:
        """Extract geoglyph data"""
        document = self.root.find(f'.//{{{self.KML_NAMESPACE}}}Document')
        if document is None:
            return []

        folders = document.findall(f'.//{{{self.KML_NAMESPACE}}}Folder')
        all_placemarks = []

        for i, folder in enumerate(folders):
            folder_name = folder.find(f'.//{{{self.KML_NAMESPACE}}}name')
            folder_name_text = folder_name.text if folder_name is not None else f"Folder_{i}"

            if folder_name_text == 'geoglyphs':
                placemarks = folder.findall(f'.//{{{self.KML_NAMESPACE}}}Placemark')
                all_placemarks.extend(placemarks)

        logging.info(f"Total geoglyphs found: {len(all_placemarks)}")
        return all_placemarks

    def extract_coordinates(self, placemark: ET.Element) -> Tuple[Optional[float], Optional[float]]:
        """Extract coordinates from placemark"""
        point = placemark.find(f'.//{{{self.KML_NAMESPACE}}}Point')
        if point is not None:
            coords = point.find(f'.//{{{self.KML_NAMESPACE}}}coordinates')
            if coords is not None:
                coord_text = coords.text.strip()
                try:
                    lon, lat = coord_text.split(',')[:2]
                    return float(lat), float(lon)
                except (ValueError, IndexError):
                    logging.warning(f"Invalid coordinate format: {coord_text}")
        return None, None

    def create_geodataframe(self) -> gpd.GeoDataFrame:
        """Create GeoDataFrame for geoglyphs"""
        placemarks = self.extract_geoglyphs()
        coordinates = []
        names = []

        for placemark in placemarks:
            lat, lon = self.extract_coordinates(placemark)
            if lat is not None and lon is not None:
                coordinates.append([lat, lon])

                name_elem = placemark.find(f'.//{{{self.KML_NAMESPACE}}}name')
                name = name_elem.text if name_elem is not None else "Unknown"
                names.append(name)

        return gpd.GeoDataFrame({
            'name': names,
            'geometry': [Point(lon, lat) for lat, lon in coordinates]
        }, crs='EPSG:4326')


def generate_south_america_coordinates(count=5):
    """Generate random latitude and longitude coordinates in South America region

    Args:
        count (int): Number of coordinate points to generate, default 5

    Returns:
        list: List of generated coordinate points
    """

    # Define latitude/longitude boundaries and weights for each country/region
    regions = {
        'brazil': {
            'lat_range': (-33.7, 5.3),
            'lon_range': (-73.9, -34.8),
            'weight': 0.6
        },
        'bolivia': {
            'lat_range': (-22.9, -9.7),
            'lon_range': (-69.6, -57.5),
            'weight': 0.08
        },
        'colombia': {
            'lat_range': (-4.2, 12.5),
            'lon_range': (-81.7, -66.9),
            'weight': 0.08
        },
        'ecuador': {
            'lat_range': (-5.0, 1.7),
            'lon_range': (-81.1, -75.2),
            'weight': 0.05
        },
        'guyana': {
            'lat_range': (1.2, 8.5),
            'lon_range': (-61.4, -56.5),
            'weight': 0.03
        },
        'peru': {
            'lat_range': (-18.3, -0.1),
            'lon_range': (-81.3, -68.7),
            'weight': 0.08
        },
        'suriname': {
            'lat_range': (1.8, 6.0),
            'lon_range': (-58.1, -53.9),
            'weight': 0.03
        },
        'venezuela': {
            'lat_range': (0.6, 12.2),
            'lon_range': (-73.4, -59.8),
            'weight': 0.08
        },
        'french_guiana': {
            'lat_range': (2.1, 5.8),
            'lon_range': (-54.6, -51.6),
            'weight': 0.02
        }
    }

    # Generate coordinates
    coordinates = []
    region_names = list(regions.keys())
    weights = [regions[region]['weight'] for region in region_names]

    for _ in range(count):
        # Select region based on weights
        selected_region = random.choices(region_names, weights=weights)[0]
        region = regions[selected_region]

        # Generate random coordinates within selected region
        lat_min, lat_max = region['lat_range']
        lon_min, lon_max = region['lon_range']

        lat = round(random.uniform(lat_min, lat_max), 6)
        lon = round(random.uniform(lon_min, lon_max), 6)

        coordinates.append([lat, lon])

    return coordinates


class MapGenerator:
    """Map generator"""

    def __init__(self):
        self.candidate_points = generate_south_america_coordinates(count=5)

    def create_interactive_map(self, tb_gdf: gpd.GeoDataFrame, gg_gdf: gpd.GeoDataFrame,
                               tb_path: str, output_path: str = 'outputs/checkpoint_2.html') -> geemap.Map:
        """Create interactive map"""
        # Calculate centroid
        centroid = gpd.GeoSeries(tb_gdf.geometry.tolist()).union_all().centroid

        # Create map
        map_obj = geemap.Map(center=(centroid.y, centroid.x), zoom=5)
        map_obj.add_basemap('SATELLITE')

        # Add layers
        map_obj.add_gdf(gg_gdf, layer_name="Geoglyphs")
        map_obj.add_kml(
            in_kml=tb_path,
            layer_name="TerraBrasilis, 2023",
            info_mode="on_hover",
            style={'color': 'red', 'weight': 2, 'fillOpacity': 0.1}
        )

        # Create candidate points data
        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in self.candidate_points],
            crs='EPSG:4326'
        )

        # Add candidate points
        map_obj.add_gdf(
            points_gdf,
            layer_name="Candidate Points",
            style={
                'color': 'yellow',
                'weight': 3,
                'fillOpacity': 0.8,
                'radius': 8
            }
        )

        # Save map
        map_obj.to_html(output_path)
        logging.info(f"Map saved to: {output_path}")
        return map_obj


class SatelliteTileGenerator:
    """Satellite tile generator"""

    def __init__(self, output_dir: str = "outputs/tiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_tile(self, lat: float, lon: float, zoom: int = 17) -> str:
        """Create single satellite tile"""
        # Create map
        m = geemap.Map(center=(lat, lon), zoom=zoom)
        m.add_basemap('SATELLITE')

        # File paths
        html_path = self.output_dir / f"tile_{lat:.6f}_{lon:.6f}.html"
        png_path = self.output_dir / f"tile_{lat:.6f}_{lon:.6f}.png"

        # Save HTML
        m.to_html(str(html_path))

        # Use Selenium for screenshot
        self._capture_screenshot(html_path, png_path)

        logging.info(f"Tile saved: {png_path}")
        return str(png_path)

    def _capture_screenshot(self, html_path: Path, png_path: Path) -> None:
        """Capture screenshot using Selenium"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=800,800")

        driver = webdriver.Chrome(options=chrome_options)
        try:
            driver.get("file://" + str(html_path.resolve()))
            time.sleep(5)  # Wait for tiles to load
            driver.save_screenshot(str(png_path))
        finally:
            driver.quit()

    def create_tiles_for_points(self, points: List[List[float]]) -> List[str]:
        """Create tiles for multiple points"""
        tile_paths = []
        for lat, lon in points:
            tile_path = self.create_tile(lat, lon)
            tile_paths.append(tile_path)
        return tile_paths


class ImageAnalyzer:
    """Image analyzer using AI for analysis"""

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """Convert image to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_tile(self, tile_path: str, prompt: str, model: str = "gpt-4o") -> str:
        """Analyze single tile"""
        try:
            image_base64 = self.image_to_base64(tile_path)
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Failed to analyze tile {tile_path}: {e}")
            return f"Analysis failed: {str(e)}"

    def analyze_all_tiles(self, prompt: str = "") -> List[Dict[str, Any]]:
        """Analyze all tiles and save results"""
        # Create results directory
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        prompt = """
        This site may potentially be an unknown archaeological site.\n
\n
Please assess the following characteristics:\n
\n
1. Geometric earthworks (plazas, mounds, causeways)\n
2. Systematic landscape organization\n
3. Evidence of pre-Columbian engineering\n
4. Integration with other network nodes\n
5. Preservation state and modern disturbance\n
\n
Provide detailed analysis of all visible archaeological features.\n
Rate confidence (0-1) for the possibility of each feature being part of an unknown archaeological site.\n
Identify specific coordinates of key elements.\n
"""

        # Get all PNG files
        tiles_dir = Path("outputs/tiles")
        png_files = list(tiles_dir.glob("*.png"))

        results = []

        for png_file in png_files:
            # Extract coordinates from filename
            match = re.search(r"tile_(-?\d+\.\d+)_(-?\d+\.\d+)\.png", png_file.name)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))

                # Analyze tile
                analysis = self.analyze_tile(str(png_file), prompt)

                # Create result entry
                result = {
                    "filename": png_file.name,
                    "latitude": lat,
                    "longitude": lon,
                    "prompt": prompt,
                    "analysis": analysis
                }

                results.append(result)

        # Save results to JSON
        output_file = results_dir / "tile_analysis_results.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"Analysis completed. Results saved to {output_file}")
        return results


class GeoAnalysisWorkflow:
    """Complete geographic analysis workflow"""

    def __init__(self):
        self.processor = GeoDataProcessor()
        self.map_generator = MapGenerator()
        self.tile_generator = SatelliteTileGenerator()
        self.image_analyzer = ImageAnalyzer()

    def run_full_analysis(self, tb_path: str = 'terrabrasilis_2023.0.kml',
                          gg_path: str = 'amazon_geoglyphs.kml') -> Dict[str, Any]:
        """Run complete analysis workflow"""
        logging.info("Starting geographic analysis workflow")

        try:
            # 1. Load TerraBrasilis data
            tb_gdf = self.processor.load_terrabrasilis_data(tb_path)

            # 2. Parse geoglyph data
            kml_parser = KMLParser(gg_path)
            gg_gdf = kml_parser.create_geodataframe()

            # 3. Create interactive map
            map_obj = self.map_generator.create_interactive_map(tb_gdf, gg_gdf, tb_path)

            # 4. Generate satellite tiles
            candidate_points = self.map_generator.candidate_points
            tile_paths = self.tile_generator.create_tiles_for_points(candidate_points)

            # 5. Analyze tiles
            analysis_results = self.image_analyzer.analyze_all_tiles()

            # Return result summary
            result_summary = {
                "terrabrasilis_features": len(tb_gdf),
                "geoglyphs_found": len(gg_gdf),
                "tiles_generated": len(tile_paths),
                "analyses_completed": len(analysis_results),
                "map_saved": "outputs/checkpoint_2.html",
                "results_saved": "outputs/results/tile_analysis_results.json"
            }

            logging.info(f"Analysis completed: {result_summary}")
            return result_summary

        except Exception as e:
            logging.error(f"Workflow execution failed: {e}")
            raise


def main():
    """Main function"""
    workflow = GeoAnalysisWorkflow()
    results = workflow.run_full_analysis()
    print(f"Analysis completed. Results saved to outputs/results/tile_analysis_results.json")
    return results


if __name__ == "__main__":
    main()