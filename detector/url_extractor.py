import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class URLExtractionError(Exception):
    """Raised when a job URL cannot be fetched or parsed."""
    pass


class JobURLExtractor:
    """
    Extract structured job information from a job posting URL.

    Output:
        title
        company_profile
        description
        requirements
        benefits
        source_url
    """

    def __init__(self, timeout=15):
        self.timeout = timeout

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }

    # =========================================================
    # URL VALIDATION
    # =========================================================

    def _validate_url(self, url):
        if not isinstance(url, str) or not url.strip():
            raise URLExtractionError("URL cannot be empty.")

        url = url.strip()
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            raise URLExtractionError(
                "URL must start with http:// or https://"
            )

        if not parsed.netloc:
            raise URLExtractionError("Invalid URL.")

        return url

    # =========================================================
    # FETCH PAGE
    # =========================================================

    def _fetch_page(self, url):
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:
                raise URLExtractionError(
                    "The URL does not appear to contain an HTML page."
                )

            # Let requests determine the encoding from the response.
            # This avoids the replacement-character issue caused by
            # reading response.raw directly.
            response.encoding = response.apparent_encoding or response.encoding

            html = response.text

            # Safety limit.
            max_size = 2 * 1024 * 1024

            if len(html.encode("utf-8", errors="ignore")) > max_size:
                html = html[:max_size]

            return html, response.url

        except URLExtractionError:
            raise

        except requests.RequestException as error:
            raise URLExtractionError(
                f"Could not fetch the URL: {error}"
            )

        except Exception as error:
            raise URLExtractionError(
                f"Unexpected error while fetching URL: {error}"
            )

    # =========================================================
    # TEXT CLEANING
    # =========================================================

    def _clean_text(self, text):
        if not text:
            return ""

        text = text.replace("\u00a0", " ")

        # Fix common HTML text-boundary issues
        text = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text
        )

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # =========================================================
    # LINKEDIN COMPANY
    # =========================================================

    def _extract_linkedin_company(self, soup):
        """
        Extract the company name from LinkedIn's job metadata.

        Example:

        Network Techlab India Limited Mumbai, Maharashtra, India
        3 days ago Over 200 applicants

        becomes:

        Network Techlab India Limited
        """

        # LinkedIn normally places the company/job metadata
        # inside an h4 element.
        for tag in soup.find_all(["h4", "h3"]):

            text = self._clean_text(
                tag.get_text(" ", strip=True)
            )

            if not text:
                continue

            lower = text.lower()

            if "applicants" not in lower:
                continue

            # Remove everything beginning with "X days ago".
            before_metadata = re.split(
                r"\s+\d+\s+days?\s+ago\b",
                text,
                maxsplit=1,
                flags=re.IGNORECASE
            )[0]

            before_metadata = self._clean_text(
                before_metadata
            )

            # Known LinkedIn location formats.
            location_pattern = (
                r"\s+"
                r"(?:Mumbai|Pune|Delhi|Bengaluru|Bangalore|"
                r"Hyderabad|Chennai|Kolkata|Noida|Gurugram|Gurgaon|"
                r"Thane|Navi Mumbai)"
                r"(?:,\s*[A-Za-z .-]+)*"
                r"(?:,\s*India)?$"
            )

            company = re.sub(
                location_pattern,
                "",
                before_metadata,
                flags=re.IGNORECASE
            )

            company = self._clean_text(company)

            if company:
                return company

        # -----------------------------------------------------
        # Fallback: "Join to apply ... at Company"
        # -----------------------------------------------------

        page_text = self._clean_text(
            soup.get_text(" ", strip=True)
        )

        match = re.search(
            r"Join to apply.*?\bat\s+(.+?)(?:"
            r"\s+\d+\s+days?\s+ago|"
            r"\s+Over\s+\d+\s+applicants|"
            r"$)",
            page_text,
            flags=re.IGNORECASE
        )

        if match:
            company = self._clean_text(
                match.group(1)
            )

            company = re.sub(
                r"\s+(?:Mumbai|Pune|Delhi|Bengaluru|Bangalore|"
                r"Hyderabad|Chennai|Kolkata|Noida|Gurugram|Gurgaon|"
                r"Thane|Navi Mumbai|Maharashtra|India)$",
                "",
                company,
                flags=re.IGNORECASE
            )

            return self._clean_text(company)

        return ""

    # =========================================================
    # LINKEDIN SECTIONS
    # =========================================================

    def _extract_linkedin_sections(self, soup):
        """
        Extract actual job responsibilities and requirements.

        LinkedIn contains a lot of unrelated content on the same
        page, so we first collect useful text candidates and then
        classify them.
        """

        candidates = []

        # -----------------------------------------------------
        # 1. Collect list items
        # -----------------------------------------------------

        for tag in soup.find_all("li"):

            text = self._clean_text(
                tag.get_text(" ", strip=True)
            )

            if not text:
                continue

            if len(text) < 20:
                continue

            if len(text) > 1000:
                continue

            candidates.append(text)

        # -----------------------------------------------------
        # 2. Collect paragraphs if necessary
        # -----------------------------------------------------

        if len(candidates) < 10:

            for tag in soup.find_all("p"):

                text = self._clean_text(
                    tag.get_text(" ", strip=True)
                )

                if not text:
                    continue

                if len(text) < 20:
                    continue

                if len(text) > 1000:
                    continue

                candidates.append(text)

        # -----------------------------------------------------
        # 3. Remove duplicates
        # -----------------------------------------------------

        unique_candidates = []
        seen = set()

        for text in candidates:

            normalized = re.sub(
                r"\s+",
                " ",
                text.lower()
            ).strip()

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_candidates.append(text)

        # -----------------------------------------------------
        # 4. Job-content classification
        # -----------------------------------------------------

        responsibility_keywords = [
            "contribute to",
            "help integrate",
            "participate in",
            "stay abreast",
        ]

        requirement_keywords = [
            "pursuing",
            "degree",
            "b.tech",
            "b.e",
            "bachelor",
            "proficiency",
            "familiarity",
            "understanding",
            "strong analytical",
            "eagerness",
            "previous internship",
            "knowledge of",
            "exposure to",
            "interest or experience",
        ]

        responsibilities = []
        requirements = []

        for text in unique_candidates:

            lower = text.lower()

            # -------------------------------------------------
            # Ignore obvious LinkedIn metadata
            # -------------------------------------------------

            if (
                "intern - software developer" in lower
                and not any(
                    keyword in lower
                    for keyword in responsibility_keywords
                )
            ):
                continue

            if "applicants" in lower:
                continue

            if "open jobs" in lower:
                continue

            if "mumbai metropolitan region" in lower:
                continue

            if re.search(
                r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b",
                lower
            ):
                continue

            # Recommended-job cards usually contain multiple
            # job titles/company/location fields.
            if (
                lower.count("intern") >= 2
                and (
                    "mumbai" in lower
                    or "navi mumbai" in lower
                    or "thane" in lower
                )
            ):
                continue

            # -------------------------------------------------
            # Requirements
            # -------------------------------------------------

            if any(
                keyword in lower
                for keyword in requirement_keywords
            ):
                requirements.append(text)
                continue

            # -------------------------------------------------
            # Responsibilities
            # -------------------------------------------------

            if any(
                keyword in lower
                for keyword in responsibility_keywords
            ):
                responsibilities.append(text)

        # -----------------------------------------------------
        # Remove accidental duplicates again
        # -----------------------------------------------------

        responsibilities = self._unique_list(
            responsibilities
        )

        requirements = self._unique_list(
            requirements
        )

        return responsibilities, requirements

    # =========================================================
    # UNIQUE LIST
    # =========================================================

    def _unique_list(self, items):

        result = []
        seen = set()

        for item in items:

            normalized = re.sub(
                r"\s+",
                " ",
                item.lower()
            ).strip()

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(item)

        return result

    # =========================================================
    # GENERIC EXTRACTION
    # =========================================================

    def _extract_generic(self, soup):

        # Remove non-content elements.
        for tag in soup([
            "script",
            "style",
            "noscript",
            "svg",
            "footer",
            "nav"
        ]):
            tag.decompose()

        title = ""

        if soup.title:
            title = self._clean_text(
                soup.title.get_text(" ", strip=True)
            )

        # Prefer h1.
        h1 = soup.find("h1")

        if h1:

            h1_text = self._clean_text(
                h1.get_text(" ", strip=True)
            )

            if h1_text:
                title = h1_text

        text_parts = []

        for tag in soup.find_all(
            ["p", "li", "div"]
        ):

            text = self._clean_text(
                tag.get_text(" ", strip=True)
            )

            if not text:
                continue

            if len(text) < 20:
                continue

            if len(text) > 1500:
                continue

            text_parts.append(text)

        text_parts = self._unique_list(
            text_parts
        )

        return {
            "title": title,
            "company_profile": "",
            "description": "\n".join(text_parts),
            "requirements": "",
            "benefits": "",
        }

    # =========================================================
    # MAIN EXTRACTION
    # =========================================================

    def extract(self, url):

        try:

            url = self._validate_url(url)

            html, final_url = self._fetch_page(url)

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            hostname = urlparse(
                final_url
            ).netloc.lower()

            # =================================================
            # LINKEDIN
            # =================================================

            if "linkedin.com" in hostname:

                # -------------------------------------------------
                # Title
                # -------------------------------------------------

                title = ""

                h1 = soup.find("h1")

                if h1:
                    title = self._clean_text(
                        h1.get_text(" ", strip=True)
                    )

                if not title and soup.title:
                    title = self._clean_text(
                        soup.title.get_text(" ", strip=True)
                    )

                # -------------------------------------------------
                # Company
                # -------------------------------------------------

                company = self._extract_linkedin_company(
                    soup
                )

                # -------------------------------------------------
                # Responsibilities / Requirements
                # -------------------------------------------------

                responsibilities, requirements = (
                    self._extract_linkedin_sections(
                        soup
                    )
                )

                description = "\n".join(
                    f"• {item}"
                    for item in responsibilities
                )

                requirements_text = "\n".join(
                    f"• {item}"
                    for item in requirements
                )

                return {
                    "title": title,
                    "company_profile": company,
                    "description": description,
                    "requirements": requirements_text,
                    "benefits": "",
                    "source_url": final_url,
                }

            # =================================================
            # GENERIC WEBSITE
            # =================================================

            data = self._extract_generic(
                soup
            )

            data["source_url"] = final_url

            return data

        except URLExtractionError:
            raise

        except Exception as error:
            raise URLExtractionError(
                f"Failed to extract job information: {error}"
            )