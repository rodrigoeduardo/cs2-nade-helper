"""
Browser automation module to navigate csnades.gg.
"""

import time
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class BrowserController:
    """Controls a browser instance to show nade lineups."""

    BASE_URL = "https://csnades.gg"

    def __init__(self) -> None:
        self.driver: Optional[webdriver.Chrome] = None
        self._setup_driver()
        # Navigate to BASE_URL on startup
        if self.driver:
            try:
                print(f"[browser] Opening {self.BASE_URL}")
                self.driver.get(self.BASE_URL)
            except Exception as e:
                print(f"[browser] Failed to open BASE_URL: {e}")

    def _setup_driver(self) -> None:
        """Initialize the Chrome driver."""
        options = Options()
        # options.add_argument("--headless")  # Keep visible for now
        options.add_argument("--window-size=1280,720")
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"[browser] Failed to start driver: {e}")

    def click_element(self, selector: str, by: str = By.CSS_SELECTOR, timeout: int = 5) -> bool:
        """
        Clicks an element specified by a selector.
        """
        if not self.driver:
            return False

        try:
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(EC.element_to_be_clickable((by, selector)))
            element.click()
            print(f"[browser] Clicked element: {selector} (by {by})")
            return True
        except Exception as e:
            # print(f"[browser] Failed to click {selector}: {e}")
            return False

    def _enable_all_nades(self) -> None:
        """
        Clicks the 'Menu' button (3 dots) on the radar and selects 'Show all'.
        """
        # 1. Click the Menu button (3 dots)
        # Found via analysis: div[aria-label="Menu"] or data-tooltip-id="map-menu"
        menu_clicked = self.click_element('div[aria-label="Menu"]', By.CSS_SELECTOR)
        if not menu_clicked:
            # Fallback: try generic aria-label or other attributes
            menu_clicked = self.click_element('button[aria-label="Menu"]', By.CSS_SELECTOR)
        
        if not menu_clicked:
            print("[browser] Could not find Radar Menu button.")
            return

        # 2. Click "Show all"
        # Usually appears in a dropdown. We search by text.
        show_all_xpath = "//*[contains(text(), 'Show all')]"
        clicked_show_all = self.click_element(show_all_xpath, By.XPATH)
        
        if clicked_show_all:
            print("[browser] Enabled 'Show all' nades on radar.")
            # Click on the body to close the dropdown
            try:
                self.driver.find_element(By.TAG_NAME, "body").click()
                print("[browser] Closed dropdown by clicking body.")
            except Exception:
                pass
        else:
            print("[browser] Could not find 'Show all' option.")

    def _hide_throw_buttons(self) -> None:
        """
        Makes all buttons that contain classes 'throw-from-t' or 'throw-from-ct' semi-transparent and unclickable.
        """
        if not self.driver:
            return

        try:
            script = """
            var buttons = document.querySelectorAll('button.throw-from-t, button.throw-from-ct, button.throw-from-any');
            var count = 0;
            buttons.forEach(function(button) {
                button.style.opacity = '0.5';
                button.style.pointerEvents = 'none';
                count++;
            });
            return count;
            """
            modified_count = self.driver.execute_script(script)
            print(f"[browser] Made {modified_count} throw buttons semi-transparent and unclickable (throw-from-t/throw-from-ct/throw-from-any).")
        except Exception as e:
            print(f"[browser] Failed to modify throw buttons: {e}")

    def navigate(self, map_name: str, grenade_type: str, callout: Optional[str] = None) -> None:
        """
        Navigate to the relevant csnades.gg page.
        URL structure example: https://csnades.gg/maps/mirage/smoke
        """
        if not self.driver:
            return

        # Normalize grenade type for URL (e.g., 'smokes', 'molotovs', 'flashbangs', 'hegrenades')
        nade_slug = grenade_type.lower()
        if "flash" in nade_slug:
            nade_slug = "flashbangs"
        elif "molotov" in nade_slug or "incendiary" in nade_slug:
            nade_slug = "molotovs"
        elif "smoke" in nade_slug:
            nade_slug = "smokes"
        elif "he" in nade_slug:
            nade_slug = "hegrenades"
        
        # Extract map name (remove 'de_' prefix if present)
        map_slug = map_name.replace("de_", "").lower()

        # URL structure: https://csnades.gg/mirage/smokes
        url = f"{self.BASE_URL}/{map_slug}/{nade_slug}"
        
        try:
            print(f"[browser] Navigating to {url}")
            self.driver.get(url)
            
            # Attempt to enable all nades on the radar
            self._enable_all_nades()
            
            # Hide throw buttons after enabling all nades
            self._hide_throw_buttons()
            
        except Exception as e:
            print(f"[browser] Navigation failed: {e}")

    def _visualize_click(self, element, x: int, y: int) -> None:
        """
        Injects a red dot at the click coordinates relative to the element.
        """
        if not self.driver:
            return
        
        # Calculate absolute position of the element to place the dot correctly
        # Note: This script assumes the element is positioned relatively or we can use absolute page coordinates.
        # A simpler approach for visual debugging is to add a fixed dot at the element's top-left + offset.
        
        # We use JavaScript to create a dot.
        # We need the element's bounding rect to know where to put the dot on the page.
        script = """
        var el = arguments[0];
        var rect = el.getBoundingClientRect();
        var dot = document.createElement('div');
        dot.style.position = 'fixed';
        dot.style.left = (rect.left + arguments[1]) + 'px';
        dot.style.top = (rect.top + arguments[2]) + 'px';
        dot.style.width = '10px';
        dot.style.height = '10px';
        dot.style.backgroundColor = 'red';
        dot.style.borderRadius = '50%';
        dot.style.zIndex = '9999';
        dot.style.pointerEvents = 'none'; // Allow clicking through
        document.body.appendChild(dot);
        
        // Remove after 1 second
        setTimeout(function() {
            dot.remove();
        }, 1000);
        """
        self.driver.execute_script(script, element, x, y)

    def click_map_position(self, relative_x: float, relative_y: float) -> None:
        """
        Clicks the map at the given relative coordinates (0.0-1.0).
        """
        if not self.driver:
            return

        try:
            # Attempt to find the main map image.
            images = self.driver.find_elements(By.TAG_NAME, "img")
            map_element = None
            
            for img in images:
                try:
                    src = img.get_property('src')

                    if 'game_radar' in src:
                        map_element = img
                        break
                except:
                    continue
            
            if not map_element:
                print("[browser] Could not identify map element.")
                return

            width = map_element.rect['width']
            height = map_element.rect['height']
            print(f"[browser] Map element size: {width}x{height}")
            
            target_x = int(width * relative_x)
            target_y = int(height * relative_y)
            
            self._visualize_click(map_element, target_x, target_y)

            # Calculate offsets from the center of the element (W3C standard compliant)
            x_offset = target_x - (width / 2)
            y_offset = target_y - (height / 2)
            
            # ActionChains to click with offset
            # move_to_element_with_offset uses offset from the CENTER of the element in newer Selenium versions
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            actions.move_to_element_with_offset(map_element, x_offset, y_offset).click().perform()
            
            print(f"[browser] Clicked map at ({relative_x:.2f}, {relative_y:.2f}) -> Offset ({x_offset:.1f}, {y_offset:.1f})")
            
        except Exception as e:
            print(f"[browser] Failed to click map position: {e}")

    def close(self) -> None:
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                print(f"[browser] Error during cleanup: {e}")
            finally:
                self.driver = None
