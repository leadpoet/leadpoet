# ========================================================================
# INDUSTRY TAXONOMY
# ========================================================================
# Dictionary keys = sub_industry (848 specific categories like "SaaS", "3D Printing")
# "industries" field = valid parent industries (50 broader categories like "Software", "Manufacturing")
# "definition" field = description of what companies in this sub_industry do
# ========================================================================

INDUSTRY_TAXONOMY = {
    "3D Printing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that build hardware, software, or services for additive manufacturing — producing physical objects from digital models layer by layer."
    },
    "3D Technology": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies that provide a real-life 3D visual appearance which is displayed in print, in a computer, or in film or television."
    },
    "3PL": {
        "industries": ['Transportation'],
        "definition": "Third-party logistics providers that handle warehousing, order fulfillment, freight management, and distribution on behalf of e-commerce brands and manufacturers."
    },
    "A/B Testing": {
        "industries": ['Data and Analytics'],
        "definition": "Companies that build software for running randomized experiments — letting product, marketing, and engineering teams compare variants and measure conversion impact."
    },
    "ADAS Calibration": {
        "industries": ['Transportation'],
        "definition": "Auto-service companies that calibrate Advanced Driver Assistance System sensors — cameras, radars, lidars — after windshield replacement, collision repair, or sensor relocation."
    },
    "AEC": {
        "industries": ['Real Estate', 'Science and Engineering'],
        "definition": "Companies in the combined Architecture, Engineering, and Construction sector — including design firms, structural and civil engineers, general contractors, and integrated project-delivery teams."
    },
    "AI Agents": {
        "industries": ['Artificial Intelligence', 'Software'],
        "definition": "Companies that build autonomous or semi-autonomous AI systems that plan, reason, and act on behalf of users — including agentic frameworks, computer-use agents, and vertical agent applications."
    },
    "AI Infrastructure": {
        "industries": ['Artificial Intelligence', 'Software', 'Hardware'],
        "definition": "Companies that provide compute, model-serving, training infrastructure, vector databases, GPUs, or orchestration platforms for building and running AI applications."
    },
    "AI Safety": {
        "industries": ['Artificial Intelligence', 'Privacy and Security'],
        "definition": "Companies that build evaluations, red-teaming services, alignment research, content moderation, or policy infrastructure for safe deployment of AI systems."
    },
    "Account-Based Marketing": {
        "industries": ['Sales and Marketing', 'Software'],
        "definition": "Companies that build software or provide services for targeting marketing programs at specific high-value accounts — including ABM platforms, intent data, and orchestration tools."
    },
    "Accounting": {
        "industries": ['Financial Services', 'Professional Services'],
        "definition": "Companies that provide bookkeeping, financial-statement preparation, audit, tax, or advisory services to businesses and individuals."
    },
    "Ad Exchange": {
        "industries": ['Advertising'],
        "definition": "Companies that develop a technology platform that facilitates the buying and selling of media advertising inventory from multiple ad networks."
    },
    "Ad Network": {
        "industries": ['Advertising'],
        "definition": "Operators of platforms that connect advertisers with publisher inventory across many websites or apps, aggregating reach and managing fill rate."
    },
    "Ad Retargeting": {
        "industries": ['Advertising'],
        "definition": "Companies that build advertising technology for re-engaging visitors who left a website without converting, via display, social, or email retargeting campaigns."
    },
    "Ad Server": {
        "industries": ['Advertising'],
        "definition": "Companies that develop web-based tools used by publishers, networks, and advertisers to help with ad management, campaign management, and ad trafficking."
    },
    "Ad Targeting": {
        "industries": ['Advertising'],
        "definition": "Companies that build technology for placing ads based on user demographics, previous browsing or buying history, or other behavioral data."
    },
    "AdTech": {
        "industries": ['Advertising', 'Software'],
        "definition": "Software companies that build infrastructure, exchanges, demand-side and supply-side platforms, measurement, and verification tools for digital advertising."
    },
    "Advanced Materials": {
        "industries": ['Manufacturing', 'Science and Engineering'],
        "definition": "Companies that produce materials designed with enhanced properties that improve on traditionally used materials."
    },
    "Adventure Travel": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that organize or supply adventure-tourism experiences involving physical exertion, exploration, or risk — including guided trekking, climbing, rafting, and expedition operators."
    },
    "Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that create and place promotional content for brands across print, broadcast, digital, social, and out-of-home channels — including agencies, networks, exchanges, and ad-tech platforms."
    },
    "Advertising Platforms": {
        "industries": ['Advertising'],
        "definition": "Companies that facilitate the relationship between advertisers and publishers by allowing advertisers to purchase ad space on a publisher's website or app."
    },
    "Advice": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that work to provide, manage, store, or facilitate advice to consumers."
    },
    "Aerospace": {
        "industries": ['Science and Engineering'],
        "definition": "Companies engaged in the research, design, manufacture, operation, or maintenance of aircraft and spacecraft."
    },
    "Aesthetics": {
        "industries": ['Health Care'],
        "definition": "Practices and providers focused on non-surgical cosmetic enhancement — injectables, energy-based devices, and skin rejuvenation."
    },
    "Affiliate Marketing": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that sell a merchant's products by signing up individuals or companies to market the company's products for a commission."
    },
    "AgTech": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that apply technology—including software, hardware, and biological innovation—to agriculture, horticulture, aquaculture, and related systems, including food production, sustainability, and the use of agricultural byproducts."
    },
    "Agriculture": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that are involved in the farming, management, production, and marketing of agricultural commodities such as livestock and crops."
    },
    "Air Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that transport person, properties, or cargo from one location to another via air travel."
    },
    "Alternative Medicine": {
        "industries": ['Health Care'],
        "definition": "Companies that provide a range of medical therapies which are not regarded as orthodox by the medical profession such as herbalism, homeopathy, and acupuncture."
    },
    "Alumni": {
        "industries": ['Education'],
        "definition": "Companies that build software or provide services for connecting alumni networks of universities, schools, and former employers — including engagement, fundraising, and mentorship platforms."
    },
    "American Football": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on American Football — including leagues, teams, broadcasters, equipment manufacturers, and venue operators."
    },
    "Amusement Park and Arcade": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that feature various attractions, such as rides, arcade games, and entertainment events for patrons."
    },
    "Analytics": {
        "industries": ['Data and Analytics'],
        "definition": "Companies that focus on the systematic computational analysis of data or statistics."
    },
    "Android": {
        "industries": ['Mobile', 'Platforms', 'Software'],
        "definition": "Companies that develop products for a mobile operating system, which is built on a modified version of the Linux kernel and other open-source software, tailored for touchscreen mobile devices."
    },
    "Angel Investment": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "An individual or groups of individuals who provide capital to a business or businesses, including startups, usually in exchange for convertible debt or ownership equity."
    },
    "Animal Feed": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that produce, distribute, or sell feed for livestock, poultry, aquaculture, and other agricultural animals."
    },
    "Animation": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that produce animated content — 2D, 3D, motion-graphics, or VFX — for film, TV, streaming, advertising, gaming, or interactive media."
    },
    "App Discovery": {
        "industries": ['Apps', 'Sales and Marketing', 'Software'],
        "definition": "Companies that build tools and platforms for helping users find new mobile or web applications — including app store optimization, search, and recommendation engines."
    },
    "App Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that focus on strategically interacting with users through an app."
    },
    "Applicant Tracking System": {
        "industries": ['Software'],
        "definition": "Companies that build software for managing the hiring funnel — job postings, candidate intake, screening, interview scheduling, offer, and onboarding handoff."
    },
    "Application Performance Management": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that manage or monitor the performance of a company's code, application dependencies, transaction times, and overall user experiences."
    },
    "Application Specific Integrated Circuit (ASIC)": {
        "industries": ['Hardware'],
        "definition": "Companies that develop integrated circuit chips customized for a particular use rather intended for general-purpose use."
    },
    "Apps": {
        "industries": ['Apps', 'Software'],
        "definition": "Companies that develop software programs which run on computers or mobile devices and are designed for end-users."
    },
    "Aquaculture": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that breed, raise, and harvest fish, shellfish, or aquatic plants in controlled freshwater or marine environments."
    },
    "Architecture": {
        "industries": ['Real Estate'],
        "definition": "Companies that employs one or more licensed architects and practices the profession of architecture."
    },
    "Archiving Service": {
        "industries": ['Administrative Services'],
        "definition": "Companies whose business model is centered around the organization and storage of physical and digital information."
    },
    "Art": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that specialize in the expression of human creative skill and imagination."
    },
    "Artificial Intelligence": {
        "industries": ['Artificial Intelligence', 'Data and Analytics', 'Science and Engineering', 'Software'],
        "definition": "Companies that use computer systems to perform tasks that normally require human intelligence by perceiving an environment and using problem-solving solutions to take actions to maximize their chances of achieving their goals."
    },
    "Asset Management": {
        "industries": ['Financial Services'],
        "definition": "Companies that aim to increase the value of their clients' financial holdings by making recommendations concerning which investments (stocks, bonds, etc.) to buy, maintain, or sell. A subset of wealth management."
    },
    "Assisted Living": {
        "industries": ['Health Care'],
        "definition": "Facilities that provide meals, housekeeping, transportation, security, physical therapy, and activities for residents."
    },
    "Assistive Technology": {
        "industries": ['Health Care'],
        "definition": "Companies that develop devices, software, or equipment for people with disabilities that have difficulty performing everyday tasks."
    },
    "Auctions": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that operate auction houses, online auction platforms, or auction software — including art, collectibles, vehicle, real-estate, and B2B liquidation auctions."
    },
    "Audio": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that build audio hardware, software, content, or streaming services — including speakers, headphones, recording equipment, audio editing tools, and audio platforms."
    },
    "Audiobooks": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that produce, distribute, or stream audiobooks — including audiobook publishers, narrator-studio production, listening apps, and audiobook subscription services."
    },
    "Augmented Reality": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies that develop technology which overlays a user's view of the real world with computer-generated images or sensations, thus combining the two environments."
    },
    "Auto Body Shop": {
        "industries": ['Transportation'],
        "definition": "Small to mid-sized facilities that perform cosmetic and structural vehicle-body repairs — including dent removal, paint refinishing, and panel replacement."
    },
    "Auto Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that provide automobile insurance to protect the consumer against financial loss in the event of an accident or theft."
    },
    "Automotive": {
        "industries": ['Transportation'],
        "definition": "Companies that design, manufacture, sell, repair, or service motor vehicles, including OEMs, parts suppliers, dealerships, and aftermarket specialists."
    },
    "Autonomous Vehicles": {
        "industries": ['Transportation'],
        "definition": "Companies that develop vehicles which can guide themselves without human conduction."
    },
    "B2B": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies whose business model focuses on selling their products and services to other businesses."
    },
    "B2B Payments": {
        "industries": ['Financial Services', 'Payments'],
        "definition": "Companies that automate business-to-business payment flows — including accounts-payable software, virtual cards, ACH, wire automation, and trade-finance platforms."
    },
    "B2C": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies whose business model is selling products and services directly to consumers."
    },
    "BIM": {
        "industries": ['Software', 'Real Estate', 'Science and Engineering'],
        "definition": "Building Information Modeling — software platforms and services for 3D-modeled, data-rich representations of buildings used across design, construction, and facility operations."
    },
    "BNPL": {
        "industries": ['Financial Services', 'Payments'],
        "definition": "Buy-Now-Pay-Later providers offering interest-free or short-term installment financing at checkout, integrated into merchant payment flows."
    },
    "Baby": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that design or sell products and services for infants and toddlers — including baby gear, apparel, food, toys, and developmental products."
    },
    "Background Check": {
        "industries": ['Professional Services', 'Software'],
        "definition": "Companies that verify employment, education, criminal, credit, and identity histories for hiring, tenant screening, and ongoing risk monitoring."
    },
    "Bakery": {
        "industries": ['Food and Beverage'],
        "definition": "Bakeries and bakery suppliers that produce or sell breads, pastries, cakes, and other baked goods — retail bakeries, wholesale bakeries, and bakery-tech platforms."
    },
    "Banking": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Financial institutions that accept deposits and channel the money into lending activities."
    },
    "Baseball": {
        "industries": ['Sports'],
        "definition": "Companies that produce goods or have their core business focus around the sport of baseball."
    },
    "Basketball": {
        "industries": ['Sports'],
        "definition": "Companies that produce goods or have their core business focus around the sport of basketball."
    },
    "Battery": {
        "industries": ['Energy'],
        "definition": "Companies that produce and research sources of electric power consisting of electrochemical cells with external connections for powering electrical devices."
    },
    "Battery Storage": {
        "industries": ['Sustainability', 'Energy'],
        "definition": "Companies that develop, manufacture, deploy, or operate energy-storage systems for grid, commercial, and residential applications — including lithium-ion and next-generation chemistries."
    },
    "Beauty": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce substances or products used to enhance the appearance of a person's face or the fragrance and texture of the body."
    },
    "Behavioral Health": {
        "industries": ['Health Care'],
        "definition": "Companies that deliver or build software for treating mental-health and substance-use disorders, often integrating clinical care with care management and population analytics."
    },
    "Big Data": {
        "industries": ['Data and Analytics'],
        "definition": "Companies that build platforms, pipelines, or services for ingesting, storing, processing, and analyzing data sets too large or complex for traditional databases."
    },
    "Billing": {
        "industries": ['Payments', 'Software'],
        "definition": "Companies that offer products or services which send an invoice to customers to pay for goods and services."
    },
    "Biofuel": {
        "industries": ['Energy', 'Natural Resources', 'Sustainability'],
        "definition": "Companies that revolve around a fuel which is produced through contemporary processes from biomass rather than a fuel produced by the very slow geological processes involved in the formation of fossil fuels such as oil."
    },
    "Bioinformatics": {
        "industries": ['Biotechnology', 'Data and Analytics', 'Science and Engineering'],
        "definition": "Companies that specialize in the application of computational technology to handle the rapidly growing repository of information related to molecular biology."
    },
    "Biomass Energy": {
        "industries": ['Energy', 'Natural Resources', 'Sustainability'],
        "definition": "Companies that develop the use of organic materials (a renewable and sustainable source of energy) to create electricity or other forms of power."
    },
    "Biometrics": {
        "industries": ['Biotechnology', 'Data and Analytics', 'Science and Engineering'],
        "definition": "Companies that develop software by which a person can be uniquely identified by evaluating one or more distinguishing biological traits."
    },
    "Biopharma": {
        "industries": ['Biotechnology', 'Health Care', 'Science and Engineering'],
        "definition": "Companies that develop pharmaceutical drugs using biological sources — including monoclonal antibodies, gene therapies, cell therapies, and other biologics produced via living systems."
    },
    "Biotechnology": {
        "industries": ['Biotechnology', 'Science and Engineering'],
        "definition": "Companies that use living organisms or biological processes to develop products and services for industrial, agricultural, medical, or other technological purposes."
    },
    "Bitcoin": {
        "industries": ['Financial Services', 'Payments', 'Software'],
        "definition": "Companies that build infrastructure, exchanges, custody, mining, payments, or financial products specifically tied to the Bitcoin protocol and asset."
    },
    "Blockchain": {
        "industries": ['Blockchain and Cryptocurrency'],
        "definition": "Companies that build distributed-ledger protocols, consensus systems, smart-contract platforms, or developer infrastructure for decentralized applications."
    },
    "Blogging Platforms": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that provide a software which allows users to publish their written content on the internet."
    },
    "Boating": {
        "industries": ['Sports'],
        "definition": "Companies involved in the leisurely activity of traveling by boat, or the recreational use of a boat."
    },
    "Body Care": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell topical products for full-body cleansing, moisturizing, and treatment — including lotions, body washes, scrubs, and body oils."
    },
    "Brand Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that work in tandem with businesses to develop a long-term, strategic plan to boost a brand's recognition and reputation among consumers."
    },
    "Brewing": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that brew, distribute, or sell beer and adjacent malt beverages — including craft breweries, mass-market breweries, brewpubs, and brewing equipment makers."
    },
    "Broadcasting": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that work to transmit audio or video content by radio or television."
    },
    "Browser Extensions": {
        "industries": ['Software'],
        "definition": "Companies that develop a small software module for customizing a web browser."
    },
    "Bug Bounty": {
        "industries": ['Privacy and Security'],
        "definition": "Companies that operate platforms or programs paying ethical hackers to find and report security vulnerabilities in clients applications and infrastructure."
    },
    "Building Maintenance": {
        "industries": ['Real Estate'],
        "definition": "Companies that perform general repairs to buildings and preventative maintenance of systems."
    },
    "Building Material": {
        "industries": ['Real Estate', 'Manufacturing'],
        "definition": "Companies that manufacture or distribute construction materials — including lumber, concrete, steel, gypsum, insulation, roofing, and finishing products."
    },
    "Business Development": {
        "industries": ['Professional Services'],
        "definition": "Companies that focus on tasks and processes to develop and implement growth opportunities within and between organizations."
    },
    "Business Information Systems": {
        "industries": ['Information Technology'],
        "definition": "Companies that develop infrastructure-related products or services to enable companies to maintain procedures or systems to collect, process, store, and distribute information within the company."
    },
    "Business Intelligence": {
        "industries": ['Data and Analytics'],
        "definition": "Companies that produce technologies, applications and practices for the collection, integration, analysis, and presentation of business information with the goal of informing operational or strategic decisions."
    },
    "Business Travel": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that arrange flights, hotels, ground transport, expense reporting, or duty-of-care services for employees traveling on business."
    },
    "CAD": {
        "industries": ['Design', 'Software'],
        "definition": "Companies that develop computer-aided design software used by architects, engineers, drafters, and product designers to create precision drawings and models."
    },
    "CI/CD": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build continuous-integration and continuous-deployment tooling for automated building, testing, and shipping of software."
    },
    "CMS": {
        "industries": ['Information Technology', 'Software'],
        "definition": "Companies that develop content management systems, a software application or set of related programs that are used to create and manage digital content. (CMS = Content Management Systems)."
    },
    "CRM": {
        "industries": ['Information Technology', 'Sales and Marketing', 'Software'],
        "definition": "Companies that create CRM software to manage customer interactions, data, and relationships for improved sales and business growth. (CRM = Customer Relationship Management)."
    },
    "Call Center": {
        "industries": ['Administrative Services'],
        "definition": "Companies that handle large volumes of telephone calls, especially for taking orders and providing customer service."
    },
    "Cannabis": {
        "industries": ['Community and Lifestyle', 'Food and Beverage', 'Health Care'],
        "definition": "Companies focused on cannabis and hemp including, but not limited to the cultivation of it, research, regulations, etc."
    },
    "Car Sharing": {
        "industries": ['Transportation'],
        "definition": "Companies that offer a peer-to-peer platform through which rental car seekers and private car owners can meet one other to do business."
    },
    "Carbon Markets": {
        "industries": ['Sustainability', 'Financial Services'],
        "definition": "Companies that develop, verify, trade, or build software for carbon credits and offsets — including registries, ratings, marketplaces, and project developers."
    },
    "Carbon Removal": {
        "industries": ['Sustainability', 'Science and Engineering'],
        "definition": "Companies that capture and durably store atmospheric carbon dioxide — including direct-air-capture, enhanced-weathering, ocean-based, and biomass-carbon-removal technologies."
    },
    "Card Issuing": {
        "industries": ['Financial Services', 'Payments'],
        "definition": "Companies that provide APIs, infrastructure, and program management for issuing branded debit, credit, virtual, and prepaid cards."
    },
    "Career Planning": {
        "industries": ['Professional Services'],
        "definition": "Companies involved in the process of helping consumers explore their interests and abilities, strategically plan their career goals, and create their future work success by designing learning and action plans to help them achieve their goals."
    },
    "Cargo Insurance": {
        "industries": ['Financial Services'],
        "definition": "Insurers and brokers that cover goods in transit by land, sea, or air — including general cargo, project cargo, and specialty commodities."
    },
    "Casino": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that operate gambling venues — including land-based casinos, online casinos, gaming resorts, and casino-management software."
    },
    "Casual Games": {
        "industries": ['Gaming'],
        "definition": "Companies that develop or publish casual video games — short-session, easy-to-learn titles played on mobile, web, or console for mass audiences."
    },
    "Catering": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that prepare and serve food at off-site events — including corporate catering, weddings, hospitality catering, and meal-program providers."
    },
    "Cause Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies focused on marketing strategies that aim to both increase profits and improve society, aligning with corporate social responsibility by incorporating activist messages into their advertising."
    },
    "Celebrity": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that manage, represent, or build platforms for celebrity talent — including talent agencies, public-relations firms, and creator-management studios."
    },
    "Charitable Foundation": {
        "industries": ['Social Impact', 'Lending and Investments'],
        "definition": "Endowed organizations and grant-making bodies that fund charitable, educational, research, or social-development programs through structured giving (e.g. Gates Foundation, Wellcome Trust)."
    },
    "Charity": {
        "industries": ['Social Impact'],
        "definition": "Non-profit organizations whose primary objectives are philanthropy and social well-being."
    },
    "Charter Schools": {
        "industries": ['Education'],
        "definition": "Schools that receive government funding but operate independently of the established state school system in which it is located."
    },
    "Chemical": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that manufacture or distribute industrial chemicals — including specialty chemicals, agrochemicals, polymers, and process intermediates for industrial and consumer applications."
    },
    "Chemical Engineering": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that specialize in the design and operation of industrial plants for chemical production."
    },
    "Child Care": {
        "industries": ['Health Care'],
        "definition": "Companies that provide care and supervision of a child or multiple children at a time."
    },
    "Children": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that develop, market, or sell products and services intended for children — including toys, apparel, education, entertainment, and child-care services."
    },
    "CivicTech": {
        "industries": ['Government and Military', 'Information Technology'],
        "definition": "Companies that enhance the relationship between the people and government with software for communications, decision-making, service delivery, and political process."
    },
    "Civil Engineering": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that design, build, and maintain physical infrastructure — including roads, bridges, tunnels, dams, water systems, and other public-works structures."
    },
    "Classifieds": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that focus on the collection of small advertisements via digital media and newspaper."
    },
    "Clean Energy": {
        "industries": ['Energy', 'Sustainability'],
        "definition": "Companies involved in creating energy that is produced through means that do not pollute the atmosphere."
    },
    "CleanTech": {
        "industries": ['Sustainability'],
        "definition": "Companies engaged in any process, product, or service that reduces negative environmental impacts through significant energy sourcing or energy efficiency improvements, the sustainable use of resources, or environmental protection activities."
    },
    "Climate Tech": {
        "industries": ['Sustainability', 'Science and Engineering'],
        "definition": "Companies that build technology to reduce greenhouse-gas emissions or adapt to climate change — including carbon removal, alternative energy, sustainable materials, and climate analytics."
    },
    "Clinical Trials": {
        "industries": ['Health Care'],
        "definition": "Companies that carry out research studies with volunteer participants to evaluate new treatments or diagnostic tests."
    },
    "Cloud Computing": {
        "industries": ['Internet Services', 'Software', 'Information Technology'],
        "definition": "Companies that provide on-demand availability of computer system resources, especially data storage and computing power, over the internet and without direct active management by the user."
    },
    "Cloud Data Services": {
        "industries": ['Information Technology', 'Internet Services'],
        "definition": "Companies that provide on-demand availability of data storage and data analysis services via the internet from a cloud computing provider's servers."
    },
    "Cloud Infrastructure": {
        "industries": ['Hardware', 'Internet Services'],
        "definition": "Companies involved in developing or maintaining the components required to provide cloud computing, such as hardware, software, servers, networking."
    },
    "Cloud Management": {
        "industries": ['Information Technology', 'Internet Services', 'Software'],
        "definition": "Companies that build tools for provisioning, monitoring, securing, optimizing cost, or governing workloads across one or more cloud providers."
    },
    "Cloud Security": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that protect data stored in cloud computing environments from theft, leakage, and deletion."
    },
    "Cloud Security Posture Management": {
        "industries": ['Privacy and Security', 'Software'],
        "definition": "Companies that build CSPM software to continuously assess cloud-environment configurations for security and compliance risks across multi-cloud deployments."
    },
    "Cloud Storage": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide software in which data is maintained, managed, backed up remotely and made available to users over a network (typically the internet)."
    },
    "Coffee": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that grow, roast, distribute, retail, or build equipment for coffee — including specialty roasters, café chains, and consumer-brand coffee products."
    },
    "Cold Chain Logistics": {
        "industries": ['Transportation'],
        "definition": "Specialized logistics providers that maintain temperature-controlled storage and transportation for perishable, pharmaceutical, or biological goods."
    },
    "Collaboration": {
        "industries": ['Collaboration'],
        "definition": "Companies that provide products or services to facilitate working with each other efficiently to create something."
    },
    "Collaborative Consumption": {
        "industries": ['Collaboration'],
        "definition": "Companies that promote the shared use of a good or services which divides the cost of purchase."
    },
    "Collectibles": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that focus on sourcing and distributing limited edition products that consumers collect."
    },
    "Collection Agency": {
        "industries": ['Administrative Services', 'Financial Services'],
        "definition": "Companies that recover delinquent debt on behalf of original creditors or buy and collect overdue receivables themselves."
    },
    "College Recruiting": {
        "industries": ['Administrative Services', 'Education'],
        "definition": "Companies that focus on recruiting and hiring recent college and university graduates."
    },
    "Collision Repair": {
        "industries": ['Transportation'],
        "definition": "Companies that repair vehicle damage from collisions — including body work, frame straightening, painting, and ADAS recalibration."
    },
    "Comics": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that publish, distribute, or sell comics, graphic novels, and webtoons — including print publishers, digital platforms, and licensing."
    },
    "Commercial Auto Insurance": {
        "industries": ['Financial Services'],
        "definition": "Insurers and brokers that underwrite policies covering vehicles used for business purposes — including delivery fleets, owner-operator trucks, and service vehicles."
    },
    "Commercial Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that offer one or more types of coverage designed to protect businesses, their owners, and their employees."
    },
    "Commercial Lending": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that provide debt-based funding arrangements between businesses and financial institutions, e.g. banks."
    },
    "Commercial Real Estate": {
        "industries": ['Real Estate'],
        "definition": "Companies that develop non-residential property used for commercial, profit-making purposes."
    },
    "Communication Hardware": {
        "industries": ['Hardware'],
        "definition": "Companies that manufacture devices which are able to transmit either analogue or digital signals over a communication cable, telephone or via wireless technology."
    },
    "Communications Infrastructure": {
        "industries": ['Hardware'],
        "definition": "Companies that provide the technology, products, and network connections which allow for the transmission of communications over large distances."
    },
    "Communities": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that specialize in specific groups of people that have particular characteristics or interests in common."
    },
    "Compensation Management": {
        "industries": ['Software'],
        "definition": "Companies that build software for benchmarking, planning, and administering employee pay — including salary, equity, bonus, and variable compensation programs."
    },
    "Compliance": {
        "industries": ['Professional Services'],
        "definition": "Companies that ensure companies and employees follow the laws, regulations, standards, and ethical practices that apply to organizations."
    },
    "Computer": {
        "industries": ['Consumer Electronics', 'Hardware'],
        "definition": "Companies that manufacture, distribute, or service computers and computer hardware — including desktops, laptops, workstations, and peripheral devices."
    },
    "Computer Vision": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies associated with the processes involved in using computers to process, interpret, and understand information from images, videos, or other visual input."
    },
    "Concerts": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies that produce technology for and platforms devoted to live music performances in front of an audience."
    },
    "Confectionery": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that produce, package, or sell candies, chocolates, and other sugar-based sweets for retail or wholesale distribution."
    },
    "Console Games": {
        "industries": ['Gaming'],
        "definition": "Companies that develop games specifically made for game play on consoles such as Xbox or Playstation."
    },
    "Construction": {
        "industries": ['Real Estate'],
        "definition": "Companies that build a wide variety of projects, including but not limited to, buildings, developments, housing, path, pavement, roads, and motorways."
    },
    "Consulting": {
        "industries": ['Professional Services'],
        "definition": "Companies that operate as a consulting agency or primarily provide specialized advisory services within their respective areas of expertise."
    },
    "Consumer Applications": {
        "industries": ['Apps', 'Software'],
        "definition": "Companies that develop or are involved with applications that are specifically tailored to meet the unique needs and preferences of customers."
    },
    "Consumer Electronics": {
        "industries": ['Consumer Electronics', 'Hardware'],
        "definition": "Companies that develop electronic devices such as TVs, smartphones, radios, or video game consoles, bought for personal rather than commercial use."
    },
    "Consumer Goods": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that manufacture or sell products purchased by individuals for personal use, spanning categories such as food, personal care, household, apparel, and electronics."
    },
    "Consumer Lending": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that originate loans to individuals — including credit-card issuers, installment lenders, BNPL providers, mortgage lenders, and consumer-lending platforms."
    },
    "Consumer Research": {
        "industries": ['Data and Analytics', 'Design'],
        "definition": "Companies that investigate the needs and opinions of consumers, with regard to a particular product or service."
    },
    "Consumer Reviews": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies involved the evaluations of a product or service made by consumers who have purchased and used, or had experience with, said products or services."
    },
    "Consumer Software": {
        "industries": ['Software'],
        "definition": "Companies that develop a class of commercial software that is sold directly to end-users as opposed to businesses."
    },
    "Contact Management": {
        "industries": ['Information Technology', 'Software'],
        "definition": "Companies that record contacts' details and track their interaction with a business."
    },
    "Container Orchestration": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build platforms for deploying, scaling, and operating containerized workloads — including Kubernetes distributions, service meshes, and container runtimes."
    },
    "Content": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies engaged in a platform in which consumers have direct access to whatever information producers wish to post."
    },
    "Content Creators": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies involved with people who produce entertaining, educational, or promotional digital media material in written, audio, or visual formats."
    },
    "Content Delivery Network": {
        "industries": ['Content and Publishing'],
        "definition": "Companies that deal specifically with a geographically distributed network of proxy servers and their data centers."
    },
    "Content Discovery": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that provide a way of allowing the creator of a video, article, or blog post to reach a much wider audience by linking their content on a more popular, highly-trafficked website."
    },
    "Content Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies focused on a strategic marketing approach through which they create and distribute valuable, relevant, and consistent content to attract and retain a clearly defined audience."
    },
    "Content Syndication": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies involved in republishing web-based content by third-party websites."
    },
    "Contests": {
        "industries": ['Gaming'],
        "definition": "Companies that build or operate competition platforms — including online contests, hackathons, sweepstakes, prize platforms, and competitive challenge marketplaces."
    },
    "Continuing Education": {
        "industries": ['Education'],
        "definition": "Companies that provide education for adults after they have left the formal education system."
    },
    "Conversion Rate Optimization": {
        "industries": ['Sales and Marketing'],
        "definition": "Agencies and software companies that improve conversion of website and app visitors via experimentation, personalization, heatmapping, and analytics."
    },
    "Cooking": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that provide products, services, content, or platforms for cooking — including recipe apps, cookware brands, cooking-class platforms, and meal-kit services."
    },
    "Corporate Training": {
        "industries": ['Education'],
        "definition": "Companies involved in the education of other companies' employees so that they improve their business-related knowledge or skills."
    },
    "Corrections Facilities": {
        "industries": ['Privacy and Security'],
        "definition": "A place that serves to confine and rehabilitate prisoners and may be classified as minimum, medium, or maximum security. Otherwise known as jail or prison."
    },
    "Cosmetic Surgery": {
        "industries": ['Health Care'],
        "definition": "Medical practices and providers that perform elective surgical procedures to enhance or alter physical appearance."
    },
    "Cosmetics": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that develop and distribute cosmetic products; mainly makeup, skincare, perfumes, and haircare products."
    },
    "Coupons": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that provide tickets or documents which can be redeemed for a financial discount or rebate when purchasing a product."
    },
    "Courier Service": {
        "industries": ['Administrative Services', 'Transportation'],
        "definition": "Companies that offer special deliveries of packages, letters, documents, or information from one place or person to another."
    },
    "Coworking": {
        "industries": ['Real Estate'],
        "definition": "Companies that provide shared workspaces and amenities to support freelancers, startups, and small businesses, fostering community and collaboration with flexible membership options."
    },
    "Craft Beer": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that produce small quantities of beer, typically emphasizing unique flavors and traditional brewing techniques."
    },
    "Creative Agency": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that plan, design, and execute creative work for brands — including advertising agencies, branding studios, and integrated creative shops."
    },
    "Creator Economy": {
        "industries": ['Sales and Marketing', 'Community and Lifestyle', 'Software', 'Media and Entertainment'],
        "definition": "Companies operating in the creator-driven monetization market — platforms, tools, agencies, and brands that enable individual creators and influencers to earn revenue from digital content, audiences, and direct-to-fan offerings."
    },
    "Credit": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that provide consumer or business credit products — including credit cards, lines of credit, credit reporting, scoring, and credit-management software."
    },
    "Credit Bureau": {
        "industries": ['Financial Services'],
        "definition": "Companies that collect, maintain, and report credit history on consumers or businesses for use in lending, insurance, and tenancy decisions."
    },
    "Credit Cards": {
        "industries": ['Financial Services', 'Lending and Investments', 'Payments'],
        "definition": "Companies that issue payment cards that allow the holder to purchase goods or services on credit."
    },
    "Cricket": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of cricket — including leagues, clubs, broadcasters, equipment manufacturers, and venue operators."
    },
    "Crowdfunding": {
        "industries": ['Financial Services'],
        "definition": "Companies that operate a platform where individuals can fund projects or ventures proposed by others in exchange for rewards, equity, or debt repayment."
    },
    "Crowdsourcing": {
        "industries": ['Collaboration'],
        "definition": "Companies that obtain information for a task by enlisting services of a large number of people via the internet."
    },
    "Cryptocurrency": {
        "industries": ['Financial Services', 'Payments', 'Software'],
        "definition": "Companies that focus on developing, issuing, and managing digital currencies which enable financial transactions over a computer network without reliance on central authorities like governments or banks to manage or sustain them."
    },
    "Customer Data Platform": {
        "industries": ['Sales and Marketing', 'Software', 'Data and Analytics'],
        "definition": "Companies that build CDP software to unify customer data from many sources into a single profile usable across marketing, sales, and service systems."
    },
    "Customer Service": {
        "industries": ['Professional Services'],
        "definition": "Companies that help businesses provide assistance with or advice about their products to their customers."
    },
    "Cyber Security": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that offer products or services that protect computer systems and digital information from intrusion, theft, damage, or disruption."
    },
    "Cycling": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport and activity of cycling — including bicycle and component manufacturers, apparel brands, race organizers, and cycling-tech platforms."
    },
    "DAO": {
        "industries": ['Blockchain and Cryptocurrency'],
        "definition": "Decentralized autonomous organizations governed by token-holder voting and smart-contract rules — includes treasury management, protocol governance, and on-chain coordination collectives."
    },
    "DEX": {
        "industries": ['Blockchain and Cryptocurrency', 'Financial Services'],
        "definition": "Decentralized exchanges that let users trade crypto assets via on-chain smart contracts without a centralized intermediary, including AMMs and orderbook DEXs."
    },
    "DIY": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that focus on the process of designing, creating or modifying any particular project or product when it is accomplished by an individual, rather than a professional. (DIY = Do-It-Yourself)."
    },
    "DRM": {
        "industries": ['Content and Publishing', 'Media and Entertainment', 'Privacy and Security'],
        "definition": "Companies that provide a set of access control technologies for restricting the use of proprietary hardware and copyrighted works. (DRM = Digital Rights Management)."
    },
    "DSP": {
        "industries": ['Hardware'],
        "definition": "Companies that develop or work with specialized microprocessor chips. (DSP = Digital Signal Processing)."
    },
    "Darknet": {
        "industries": ['Internet Services'],
        "definition": "Companies that build privacy-preserving networks, anonymization tools, or threat-intelligence services focused on the hidden web layer reachable only through specialized clients."
    },
    "Data Catalog": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build software for organizing, governing, and discovering enterprise data assets — including metadata management, lineage, and access control."
    },
    "Data Center": {
        "industries": ['Hardware', 'Information Technology'],
        "definition": "Companies engaged in a large group of networked computer servers, typically used by organizations for the remote storage, processing, or distribution of large amounts of data."
    },
    "Data Center Automation": {
        "industries": ['Hardware', 'Information Technology', 'Software'],
        "definition": "Companies that focus on the process of managing and automating the workflow and processes of a data center facility."
    },
    "Data Engineering": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build tools, services, or consulting for moving, transforming, and modeling data at scale — including ETL, orchestration, and pipeline reliability."
    },
    "Data Integration": {
        "industries": ['Data and Analytics', 'Information Technology', 'Software'],
        "definition": "Companies that specialize in providing a unified view of combined data from multiple sources or formats."
    },
    "Data Lake": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build platforms for storing large volumes of raw and semi-structured data, with open table formats and downstream analytic and ML access."
    },
    "Data Mining": {
        "industries": ['Data and Analytics', 'Information Technology'],
        "definition": "Companies that work to extract information or discover patterns from large raw data sets in order to transform the information into a comprehensible structure for further use."
    },
    "Data Observability": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build software for monitoring data-pipeline reliability — detecting freshness, schema, volume, distribution, and quality issues in production data systems."
    },
    "Data Pipelines": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build ELT, ETL, and streaming-data-pipeline tooling for ingesting and transforming data between sources and warehouses."
    },
    "Data Storage": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies that produce storage hardware, build cloud-storage services, or develop software for managing, archiving, and retrieving digital information at scale."
    },
    "Data Visualization": {
        "industries": ['Data and Analytics', 'Design', 'Information Technology', 'Software'],
        "definition": "Companies that build software, libraries, or platforms for transforming structured data into interactive charts, dashboards, and graphical reports."
    },
    "Data Warehouse": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build cloud or on-premise analytic data warehouses optimized for storing and querying structured data at scale."
    },
    "Database": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that provide a data structure that stores organized information."
    },
    "Dating": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that provide specific mechanisms for online dating through the use of internet connected personal computers or mobile devices."
    },
    "DeFi": {
        "industries": ['Blockchain and Cryptocurrency', 'Financial Services'],
        "definition": "Decentralized-finance companies that build lending, borrowing, trading, derivatives, or yield protocols operating on public blockchains via smart contracts."
    },
    "Debit Cards": {
        "industries": ['Financial Services', 'Payments'],
        "definition": "Companies that produce payment cards which deduct money directly from a consumer's checking account to pay for a purchase."
    },
    "Delivery": {
        "industries": ['Administrative Services', 'Transportation'],
        "definition": "Companies that deliver packages, letters, food, or goods — including courier services, parcel carriers, food-delivery platforms, and on-demand delivery operators."
    },
    "Dental": {
        "industries": ['Health Care'],
        "definition": "Companies that provide dental care services, products, or equipment, often including treatments, preventive care, and oral hygiene products."
    },
    "Dermatology": {
        "industries": ['Health Care'],
        "definition": "Medical practices specializing in the diagnosis and treatment of skin, hair, and nail conditions, including both clinical and cosmetic dermatology."
    },
    "Desktop Apps": {
        "industries": ['Software'],
        "definition": "Companies that develops software whose main use is as an application stored on your desktop."
    },
    "DevOps": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build tools or provide services for software-delivery automation — CI/CD, infrastructure-as-code, deployment, environment management, and developer productivity."
    },
    "Developer APIs": {
        "industries": ['Software'],
        "definition": "Companies whose primary product is an API — software intermediaries that let third-party developers integrate functionality such as payments, messaging, data, identity, or machine learning into their own applications."
    },
    "Developer Platform": {
        "industries": ['Software'],
        "definition": "Companies that focus on offering tools for developers to build upon their products or helping customers offer these tools."
    },
    "Developer Tools": {
        "industries": ['Software'],
        "definition": "Companies that build programs allowing developers to create, test, and debug software."
    },
    "Diabetes": {
        "industries": ['Health Care'],
        "definition": "Companies that develop drugs, devices, software, or services for preventing, diagnosing, monitoring, or treating diabetes — including CGM, insulin delivery, and digital-health platforms."
    },
    "Dietary Supplements": {
        "industries": ['Food and Beverage', 'Health Care'],
        "definition": "Companies that manufacture products intended to supplement the diet when taken by mouth as a pill, capsule, tablet, or liquid."
    },
    "Digital Entertainment": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce or distribute entertainment content delivered via electronic devices — including video games, streaming, digital comics, and online gambling."
    },
    "Digital Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that focus on the marketing of products or services using digital technologies on the internet, through mobile phone apps, display advertising, and any other digital mediums."
    },
    "Digital Media": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce digitized content in text, audio, video, or graphical formats which can be transmitted over the internet or computer networks."
    },
    "Digital Signage": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that produce display hardware, content-management software, or installation services for advertising and informational screens in public and commercial spaces."
    },
    "Digital Therapeutics": {
        "industries": ['Health Care', 'Software'],
        "definition": "Companies that develop evidence-based software-as-medicine to prevent, manage, or treat disease — often FDA-cleared and prescribed alongside or in place of drugs."
    },
    "Direct Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies dealing with a promotional method that involves presenting information about a company, product or service to target customers, without the use of an advertising middleman."
    },
    "Direct-to-Consumer": {
        "industries": ['Commerce and Shopping', 'Consumer Goods'],
        "definition": "Companies that sell branded consumer goods straight to end customers — bypassing wholesale and retail intermediaries — typically through their own e-commerce, social, and retail channels."
    },
    "Distillery": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that distill spirits — including whiskey, vodka, gin, rum, tequila, and other distilled alcoholic beverages, as both producers and brand operators."
    },
    "Diving": {
        "industries": ['Sports'],
        "definition": "Companies that offer services related to underwater diving, including training, equipment rental, and guided dive tours."
    },
    "Document Management": {
        "industries": ['Information Technology', 'Software'],
        "definition": "Companies that produce software to store, manage, and track electronic documents or electronic images of paper-based information captured by a document scanner."
    },
    "Document Preparation": {
        "industries": ['Administrative Services'],
        "definition": "Companies that help individuals prepare specific documents manually or via automation."
    },
    "Domain Registrar": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide a place to register or store registrations of domains."
    },
    "Drone Management": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies that build software for operating drone fleets — including flight planning, regulatory compliance, data capture, and enterprise drone operations."
    },
    "Drones": {
        "industries": ['Consumer Electronics', 'Consumer Goods', 'Hardware'],
        "definition": "Companies that focus on unmanned aerial vehicles or flying robots without a human pilot."
    },
    "E-Commerce": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that focus on facilitating the buying and selling of goods and services over an electronic network. Excludes companies that incorporate online sales as only a part of their business."
    },
    "E-Commerce Platforms": {
        "industries": ['Commerce and Shopping', 'Internet Services'],
        "definition": "Companies that develop software applications which allow online businesses to manage their website, marketing, sales, and operations."
    },
    "E-Learning": {
        "industries": ['Education', 'Software'],
        "definition": "Companies that are involved in helping individuals learn by utilizing electronic technologies to access educational curriculum in digital formats and outside of a traditional classroom."
    },
    "E-Signature": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that develop technology for attaching verification symbols to electronically transmitted documents, confirming the sender's intent to sign."
    },
    "EBooks": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that publish, distribute, sell, or build reading platforms for book content delivered in digital form to e-readers, tablets, or mobile devices."
    },
    "EV Charging": {
        "industries": ['Transportation', 'Sustainability'],
        "definition": "Companies that build, operate, or supply electric-vehicle charging infrastructure — including hardware manufacturers, network operators, and charging-management software."
    },
    "EdTech": {
        "industries": ['Education', 'Software'],
        "definition": "Companies that develop or focus on technological tools or software designed to enhance learning, whether in physical classroom, online, or blended settings."
    },
    "Edge Computing": {
        "industries": ['Software', 'Information Technology', 'Hardware'],
        "definition": "Companies that build hardware, software, or services for processing data near its source — at edge gateways, IoT devices, or regional micro-data-centers."
    },
    "Ediscovery": {
        "industries": ['Internet Services'],
        "definition": "Companies that specialize in identifying, collecting, and producing electronically stored information for legal processes, such as litigation, investigations, and compliance matters."
    },
    "Education": {
        "industries": ['Education'],
        "definition": "Companies that facilitate the process of receiving or giving systematic instruction."
    },
    "Edutainment": {
        "industries": ['Education', 'Media and Entertainment'],
        "definition": "Companies that produce technologies, software products (i.e. video games, movies, or shows), and content aimed at learning through fun mediums."
    },
    "Elderly": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that develop or sell products and services targeted at older adults — including senior-care, mobility, hearing aids, and silver-economy brands."
    },
    "Electric Vehicle": {
        "industries": ['Transportation'],
        "definition": "Companies that develop, manufacture, or supply electric vehicles and EV components — including passenger EVs, commercial EVs, batteries, drivetrains, and EV-specific infrastructure."
    },
    "Electrical Distribution": {
        "industries": ['Energy'],
        "definition": "Companies that operate or supply the electricity distribution network — local power lines, transformers, and grid edge equipment that delivers electricity to customers."
    },
    "Electronic Design Automation (EDA)": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies that produce software tools for designing electronic systems such as integrated circuits and printed circuit boards."
    },
    "Electronic Health Record (EHR)": {
        "industries": ['Health Care'],
        "definition": "Companies that keep a systemized collection of electronically-stored patient health information in a digital format."
    },
    "Electronics": {
        "industries": ['Consumer Electronics', 'Hardware'],
        "definition": "Companies that design, manufacture, or distribute electronic components, consumer devices, and industrial electronic systems."
    },
    "Email": {
        "industries": ['Information Technology', 'Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that provide email-related products or services to consumers or businesses (i.e. email service providers, newsletter platforms, email marketing services, webmail plugins, or email monitoring tools)."
    },
    "Email Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that allow others to use email to promote products and/or services."
    },
    "Embedded Finance": {
        "industries": ['Financial Services', 'Software'],
        "definition": "Companies that provide APIs and infrastructure letting non-financial businesses embed banking, payments, lending, insurance, or investing into their own products."
    },
    "Embedded Software": {
        "industries": ['Software'],
        "definition": "Companies that produce computer software which has been written to control machines or devices."
    },
    "Embedded Systems": {
        "industries": ['Hardware', 'Science and Engineering', 'Software'],
        "definition": "Companies that produce a controller with a dedicated function within a larger mechanical system."
    },
    "Emergency Medicine": {
        "industries": ['Health Care'],
        "definition": "Medical practices, services, and software focused on the diagnosis and treatment of acute illness and traumatic injury in emergency settings."
    },
    "Emerging Markets": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that invest in or do business in emerging markets, including those in developing countries."
    },
    "Employee Benefits": {
        "industries": ['Administrative Services'],
        "definition": "Companies that provide diverse benefits, including financial, health, and welfare programs, along with perks, to enhance employee well-being and satisfaction."
    },
    "Employee Experience": {
        "industries": ['Software', 'Professional Services'],
        "definition": "Companies that build software for measuring and improving the employee journey — engagement surveys, recognition, internal communications, and people analytics."
    },
    "Employment": {
        "industries": ['Professional Services'],
        "definition": "Companies that match individuals seeking jobs with organizations looking to hire, facilitating the recruitment process."
    },
    "Energy": {
        "industries": ['Energy'],
        "definition": "Companies that generate, transmit, distribute, store, or trade energy — including utilities, renewables, oil and gas, and energy software."
    },
    "Energy Efficiency": {
        "industries": ['Energy', 'Sustainability'],
        "definition": "Companies that provide products or services that aim to reduce the amount of energy required to perform a task."
    },
    "Energy Management": {
        "industries": ['Energy'],
        "definition": "Companies that build hardware, software, or services for monitoring and optimizing energy generation, distribution, consumption, and storage within buildings or grids."
    },
    "Energy Storage": {
        "industries": ['Energy'],
        "definition": "Companies that manufacture or conduct work relating to how energy can be stored."
    },
    "Enterprise Applications": {
        "industries": ['Apps', 'Software'],
        "definition": "Companies that develop a large software system platform designed to operate in a corporate environment such as a business or the government."
    },
    "Enterprise Resource Planning (ERP)": {
        "industries": ['Software'],
        "definition": "Companies that develop integrated software suites unifying finance, supply chain, HR, manufacturing, and other core business processes on a single platform."
    },
    "Enterprise Software": {
        "industries": ['Software'],
        "definition": "Companies that develop software tools designed primarily for organization-sized use rather than for individuals."
    },
    "Environmental Consulting": {
        "industries": ['Professional Services'],
        "definition": "Companies that offer a form of compliance consulting in which the consultant ensures that the client maintains an appropriate measure of compliance with environmental regulations."
    },
    "Environmental Engineering": {
        "industries": ['Science and Engineering', 'Sustainability'],
        "definition": "Companies that use scientific and engineering strategies to improve the quality of the environment, and to protect the health of living organisms from adverse environmental effects, such as pollution."
    },
    "Equestrian": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that offer products and services related to horse riding, including the sale of horses, riding gear, training, and horse care facilities."
    },
    "Estate Agent": {
        "industries": ['Real Estate'],
        "definition": "UK-style real-estate agencies that broker the sale of residential or commercial properties, equivalent to US real-estate brokerages."
    },
    "Ethereum": {
        "industries": ['Blockchain and Cryptocurrency'],
        "definition": "Companies that develop applications or services based on the Ethereum blockchain, focusing on smart contracts and decentralized applications."
    },
    "Event Management": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies that organize and coordinate events, handling logistics, venue selection, and execution of corporate, public, or private gatherings such as festivals, conferences, ceremonies, weddings, etc."
    },
    "Event Promotion": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies that focus on the process of driving business goals for a product, brand, or service through in-person events."
    },
    "Events": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies that produce, host, or build software for large-scale events — including conferences, festivals, trade shows, and virtual experiences."
    },
    "Extermination Service": {
        "industries": ['Administrative Services'],
        "definition": "Companies that provide pest-control services — eliminating insects, rodents, and other unwanted animals from residential, commercial, and agricultural properties."
    },
    "Eyewear": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that design, manufacture, or sell prescription glasses, sunglasses, contact lenses, and adjacent vision-correction or fashion eyewear products."
    },
    "Facebook": {
        "industries": ['Platforms'],
        "definition": "Companies whose primary product integrates with or builds on the Facebook/Meta platform — including Meta-ads tools, page-management software, and social-analytics products."
    },
    "Facial Recognition": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that work in the creation, improvement, sharing, and/or implementation of facial recognition software."
    },
    "Facilities Support Services": {
        "industries": ['Administrative Services'],
        "definition": "Companies that provide a combination of services to support operations within facilities, including but not limited to, janitorial, maintenance, trash disposal, guard and security, mail routing, reception, and laundry."
    },
    "Facility Management": {
        "industries": ['Real Estate'],
        "definition": "Companies that specialize in a professional management discipline focused on the efficient and effective delivery of logistics and other support services related to real property and buildings."
    },
    "Family": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that develop or sell products and services targeted at families with children — including family-friendly entertainment, child-care services, and household-focused brands."
    },
    "Family Medicine": {
        "industries": ['Health Care'],
        "definition": "Primary-care medical practices providing continuous and comprehensive health care for individuals and families across all ages."
    },
    "Fantasy Sports": {
        "industries": ['Gaming', 'Sports'],
        "definition": "Companies that develop games, often played virtually, where participants assemble teams of real-life players of a professional sport."
    },
    "Farmers Market": {
        "industries": ['Food and Beverage'],
        "definition": "Food markets at which local farmers sell fruit, vegetables, and oftentimes meat, cheese, and bakery products, directly to consumers."
    },
    "Farming": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies involved in the farming of agricultural commodities such as livestock and crops."
    },
    "Fashion": {
        "industries": ['Clothing and Apparel', 'Design'],
        "definition": "Companies that design, produce, distribute, or retail apparel and accessories — including luxury, mass-market, fast-fashion, and direct-to-consumer brands."
    },
    "Fast-Moving Consumer Goods": {
        "industries": ['Consumer Goods', 'Real Estate'],
        "definition": "Companies that focus on products that are sold quickly and at a relatively low cost."
    },
    "Feature Flags": {
        "industries": ['Software'],
        "definition": "Companies that build software for controlled feature releases — runtime toggles, percentage rollouts, A/B targeting, and experimentation analytics."
    },
    "Ferry Service": {
        "industries": ['Transportation'],
        "definition": "Companies that provide transportation to consumers via large ferry boats."
    },
    "Fertility": {
        "industries": ['Health Care'],
        "definition": "Companies that provide medical services and treatments aimed at assisting individuals and couples in achieving pregnancy."
    },
    "Field Support": {
        "industries": ['Professional Services'],
        "definition": "Companies that address tasks such as the testing, documenting, setting up, and installation of network hardware."
    },
    "Field-Programmable Gate Array (FPGA)": {
        "industries": ['Hardware'],
        "definition": "Companies that develop and produce an integrated circuit designed to be configured by a customer or a designer after manufacturing."
    },
    "File Sharing": {
        "industries": ['Software'],
        "definition": "Companies involved in the practice of sharing or offering access to digital information or resources including documents, multimedia, graphics, computer programs, images, and e-books."
    },
    "Film": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that produce, distribute, or supply services for motion pictures — including studios, production companies, post-production, and film financing."
    },
    "Film Distribution": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that distribute films to theaters, streaming, broadcast, and home-video channels — including studio distribution arms and independent distributors."
    },
    "Film Production": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that are involved in the production process of a motion picture."
    },
    "FinTech": {
        "industries": ['Financial Services'],
        "definition": "Companies that use technology such as the internet, mobile devices, blockchains, or cloud computing to compete with traditional finance methods in the delivery of financial services."
    },
    "Finance": {
        "industries": ['Financial Services'],
        "definition": "Companies broadly involved in the finance sector, whose primary services relate to money, currency, or capital assets."
    },
    "Financial Exchanges": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that provide a marketplace where securities, commodities, derivatives, and other financial instruments are traded."
    },
    "Financial Services": {
        "industries": ['Financial Services'],
        "definition": "Companies that provide professional services in the finance industry, involving the investment, lending, insurance, and management of money and assets."
    },
    "First Aid": {
        "industries": ['Health Care'],
        "definition": "Companies that provide first-aid training, supplies, kits, or on-site emergency-response services for workplaces, schools, and public venues."
    },
    "Fitness": {
        "industries": ['Sports'],
        "definition": "Companies that promote physical health and well-being, especially through physical exercise."
    },
    "Flash Sale": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that help consumers find sales of goods at greatly reduced prices for short periods of time."
    },
    "Flash Storage": {
        "industries": ['Hardware'],
        "definition": "Companies that produce or provide a type of drive or system that uses flash memory to keep data for an extended period of time."
    },
    "Fleet Management": {
        "industries": ['Transportation'],
        "definition": "Companies that build software or provide services for operating commercial vehicle fleets — including telematics, dispatch, fuel, maintenance, and driver-compliance tools."
    },
    "Flowers": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that grow, distribute, sell, or arrange flowers — including florists, flower-delivery services, wholesale growers, and floral-design studios."
    },
    "Food Delivery": {
        "industries": ['Food and Beverage', 'Transportation'],
        "definition": "Companies that deliver prepared meals from restaurants to consumers — including delivery marketplaces, restaurant-owned delivery, and ghost-kitchen operators."
    },
    "Food Processing": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that transform raw agricultural products into processed and packaged food — including milling, canning, freezing, fermenting, and prepared-food manufacturing."
    },
    "Food Trucks": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that operate mobile food businesses — including individual food trucks, multi-truck brands, and food-truck management or marketplace software."
    },
    "Food and Beverage": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that research, produce, distribute, or serve consumer food or beverage products."
    },
    "Forestry": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that create, manage, use, conserve, and repair forests, woodlands, and associated resources for human and environmental benefits."
    },
    "Fossil Fuels": {
        "industries": ['Energy', 'Natural Resources'],
        "definition": "Companies engaged in the extraction, production, refining, or selling of coal, oil, or natural gas."
    },
    "Foundation Models": {
        "industries": ['Artificial Intelligence', 'Software'],
        "definition": "Companies that train or commercialize large, general-purpose AI models that serve as the base layer for downstream applications across modalities."
    },
    "Foundries": {
        "industries": ['Manufacturing'],
        "definition": "Companies that cast metal — including iron, steel, aluminum, and specialty alloy foundries serving automotive, industrial, construction, and consumer-goods markets."
    },
    "Franchise": {
        "industries": ['Commerce and Shopping', 'Community and Lifestyle'],
        "definition": "Companies operating multi-unit franchise networks or providing franchise-specific services — franchisors selling rights to operate branded units, multi-unit franchisees, and franchise-management software."
    },
    "Fraud Detection": {
        "industries": ['Financial Services', 'Payments', 'Privacy and Security'],
        "definition": "Companies that prevent and protect companies or individuals against fraud."
    },
    "Freelance": {
        "industries": ['Professional Services'],
        "definition": "Companies that build marketplaces, software, or services for freelancers and independent contractors — including freelance platforms, payment, contracting, and benefits."
    },
    "Freight Service": {
        "industries": ['Transportation'],
        "definition": "Companies that specialize in the moving of freight or cargo from one place to another."
    },
    "Fruit": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that grow, distribute, process, or sell fresh and processed fruit — including orchards, fruit distributors, juice producers, and fruit-based consumer brands."
    },
    "Fuel": {
        "industries": ['Energy'],
        "definition": "Companies engaged in petroleum, a naturally occurring liquid found beneath the earth's surface that can be refined into fuel."
    },
    "Fuel Cell": {
        "industries": ['Energy'],
        "definition": "Companies that develop fuel cells, which use chemical energy from hydrogen or other fuels to cleanly and efficiently produce electricity."
    },
    "Funding Platform": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that produce software to facilitate the acceptance of money from individuals or companies for the purpose of raising money to contribute to a project, business, or need."
    },
    "Funerals": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that provide services related to organizing and conducting funeral ceremonies, including preparation of the deceased, provision of caskets or urns, and arrangement of the funeral service."
    },
    "Furniture": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that design, manufacture, distribute, or sell furniture for residential, commercial, hospitality, and office uses."
    },
    "GPS": {
        "industries": ['Hardware', 'Navigation and Mapping'],
        "definition": "Companies that develop global positioning satellite systems. (GPS = Global Positioning Satellites)."
    },
    "GPU": {
        "industries": ['Hardware'],
        "definition": "Companies that produce a graphics processing unit which is capable of rendering graphics for display on an electronic device. (GPU = Graphics Processing Unit)."
    },
    "Gambling": {
        "industries": ['Gaming'],
        "definition": "Companies that specialize in the activity of playing games of chance for money or other stakes."
    },
    "Gamification": {
        "industries": ['Gaming'],
        "definition": "Companies that use elements of game design, mechanics, and principles in non-game contexts to engage and motivate people to achieve certain goals, learn new skills, or change their behavior."
    },
    "Gaming": {
        "industries": ['Gaming'],
        "definition": "Companies that develop, publish, or operate video games and gaming services — including studios, publishers, esports organizations, and platforms for player engagement and monetization."
    },
    "Generative AI": {
        "industries": ['Artificial Intelligence', 'Software'],
        "definition": "Companies that build or deploy generative-AI models and applications — text, image, audio, video, code generation — including model labs, foundation-model platforms, and vertical generative tools."
    },
    "Genetics": {
        "industries": ['Biotechnology', 'Health Care'],
        "definition": "Companies that conduct or commercialize research, sequencing, testing, therapy, or editing of genes and genomes for clinical, agricultural, or consumer applications."
    },
    "Geospatial": {
        "industries": ['Data and Analytics', 'Navigation and Mapping'],
        "definition": "Companies that develop, manufacture, research, and employ geospatial technology, which is used to collect, analyze and store geographic information."
    },
    "Gift": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that specialize in the creation, distribution, or retail of items intended to be given to others as presents, often featuring a wide range of products suitable for various occasions and recipients."
    },
    "Gift Card": {
        "industries": ['Commerce and Shopping', 'Financial Services'],
        "definition": "Companies that relate primarily to prepaid stored-value money cards which are used as an alternative to cash for purchases, usually within a particular store or related businesses."
    },
    "Gift Exchange": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies involved in the transfer of goods or services that although regarded as voluntary by the people involved is part of the expected social behavior."
    },
    "Gift Registry": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies housing lists of items that consumers wish to receive, and gift-givers can purchase."
    },
    "Golf": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of golf — including course operators, equipment manufacturers, apparel brands, instruction services, and tournament broadcasters."
    },
    "Google": {
        "industries": ['Platforms'],
        "definition": "Companies whose primary product builds on Google services — including Google Ads tools, Workspace integrations, Google Cloud partners, and SEO/SEM software for Google search."
    },
    "Google Glass": {
        "industries": ['Consumer Electronics', 'Hardware', 'Mobile', 'Platforms'],
        "definition": "Augmented-reality and smart-eyewear hardware and software, originally pioneered by Google Glass and now spanning industrial, medical, and consumer AR-glasses products."
    },
    "GovTech": {
        "industries": ['Government and Military', 'Information Technology'],
        "definition": "Software companies that build technology for government agencies — including digital services, citizen engagement, regulatory compliance, procurement, and public-sector workflow tools."
    },
    "Government": {
        "industries": ['Government and Military'],
        "definition": "Government agencies, government-owned enterprises, and contractors that provide products or services primarily to public-sector buyers."
    },
    "Graphic Design": {
        "industries": ['Design'],
        "definition": "Companies that create visual communication materials for their clients."
    },
    "Green Building": {
        "industries": ['Real Estate', 'Sustainability'],
        "definition": "Companies that apply environmentally responsible and resource-efficient processes throughout a building's life-cycle."
    },
    "Green Consumer Goods": {
        "industries": ['Consumer Goods', 'Sustainability'],
        "definition": "Companies that produce goods that have been made in a way that protects the natural environment."
    },
    "GreenTech": {
        "industries": ['Sustainability'],
        "definition": "Companies that focus on the use of technology or technological processes to reduce negative impacts on the environment, conserve resources, and promote sustainability."
    },
    "Grocery": {
        "industries": ['Food and Beverage'],
        "definition": "Retailers that sell food and household staples — including supermarkets, neighborhood grocers, specialty food stores, and online grocery delivery services."
    },
    "Group Buying": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that aggregate consumer demand to offer products or services at reduced prices, typically facilitated through online platforms where deals are activated once a minimum number of purchases is made."
    },
    "Growth Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Marketing agencies, consultancies, and software companies focused on data-driven, experiment-led growth — acquisition, activation, retention, referral, revenue optimization."
    },
    "Guides": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that provide guided experiences — tour guides, in-app product guides, expert-led learning, and concierge services."
    },
    "HR Tech": {
        "industries": ['Software', 'Professional Services', 'Apps'],
        "definition": "Software companies building tools for hiring, onboarding, performance management, learning, engagement, and people analytics."
    },
    "HVAC": {
        "industries": ['Physical Infrastructure'],
        "definition": "Companies that design, install, and service heating, ventilation, and air-conditioning systems for residential, commercial, and industrial buildings."
    },
    "Hair Care": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell products for cleansing, conditioning, styling, and treating hair — including shampoos, conditioners, color, and styling tools."
    },
    "Handmade": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell goods made by hand — including artisanal goods marketplaces, craft brands, and small-batch makers."
    },
    "Hardware": {
        "industries": ['Hardware'],
        "definition": "Companies that develop physical components of any computer or telecommunications system."
    },
    "Headless Commerce": {
        "industries": ['Commerce and Shopping', 'Software'],
        "definition": "Companies that build commerce platforms decoupling the frontend storefront from backend commerce engines via APIs — enabling custom storefronts across web, mobile, and emerging channels."
    },
    "Health Care": {
        "industries": ['Health Care'],
        "definition": "Companies that deliver, support, or finance medical services — including providers, payers, hospitals, clinics, devices, pharmaceuticals, and digital health platforms."
    },
    "Health Diagnostics": {
        "industries": ['Health Care'],
        "definition": "Companies engaged in the process of determining which disease or condition explains a person's symptoms and signs."
    },
    "Health Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that offer a type of insurance coverage which pays for medical, surgical, and sometimes dental expenses by the insured."
    },
    "HealthTech": {
        "industries": ['Health Care', 'Software'],
        "definition": "Software, data, and platform companies building technology for the health-care industry — including patient engagement, clinical workflow, remote monitoring, and care coordination."
    },
    "Hedge Funds": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Privately-owned companies that pool investors' dollars and reinvest them into complicated financial instruments."
    },
    "Higher Education": {
        "industries": ['Education'],
        "definition": "Companies that operate, support, or supply universities, colleges, and other institutions awarding post-secondary degrees and credentials."
    },
    "Hockey": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of hockey — including leagues, teams, broadcasters, equipment manufacturers, and rink operators."
    },
    "Home Decor": {
        "industries": ['Real Estate'],
        "definition": "Companies that design, manufacture, or retail decorative furnishings, textiles, lighting, wall art, and accessories for residential interiors."
    },
    "Home Fragrance": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell scented products for residential use — including candles, diffusers, room sprays, and incense."
    },
    "Home Health Care": {
        "industries": ['Health Care'],
        "definition": "Companies that provide health care services in a person's home for illness or injury."
    },
    "Home Improvement": {
        "industries": ['Real Estate'],
        "definition": "Companies that sell tools, materials, fixtures, or services for residential renovation, repair, remodeling, and DIY projects."
    },
    "Home Renovation": {
        "industries": ['Real Estate'],
        "definition": "Companies that focus on making improvements to an existing building or home in order to restore the building to a good state of repair."
    },
    "Home Services": {
        "industries": ['Real Estate'],
        "definition": "Companies involved with maintenance, repair, or improvement services for residential properties, such as plumbing, electrical work, cleaning, and landscaping."
    },
    "Home and Garden": {
        "industries": ['Real Estate'],
        "definition": "Companies that supply products, services, or media for home interiors and gardens — including home-improvement retailers, lawn-and-garden brands, and home-and-garden media."
    },
    "Homeland Security": {
        "industries": ['Privacy and Security'],
        "definition": "Companies that focus on the national effort to ensure a homeland is safe, secure, resilient against terrorism and other hazards."
    },
    "Homeless Shelter": {
        "industries": ['Social Impact'],
        "definition": "Service agencies that provide temporary residence for homeless individuals and families."
    },
    "Horticulture": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies engaged in plant cultivation for food, medicine, or aesthetic use — including fruit, vegetable, herb, ornamental-plant, and nursery operators."
    },
    "Hospital": {
        "industries": ['Health Care'],
        "definition": "Companies that provide comprehensive medical care, diagnostics, and treatment services to patients in a dedicated facility that is staffed by medical professionals and capable of housing patients during medical care."
    },
    "Hospitality": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that provide accommodation, food and beverage services, and other amenities to travelers and guests, often through hotels, resorts, restaurants, and related venues."
    },
    "Hotel": {
        "industries": ['Travel and Tourism'],
        "definition": "Establishments providing accommodations, meals, and other services for travelers and tourists."
    },
    "Housekeeping Service": {
        "industries": ['Administrative Services'],
        "definition": "Companies that provide residential cleaning, laundry, and household-management services on a one-time, recurring, or live-in basis."
    },
    "Human Computer Interaction": {
        "industries": ['Design', 'Science and Engineering'],
        "definition": "Companies that research the design and use of computer technology focused on the interfaces between people and computers."
    },
    "Human Resources": {
        "industries": ['Administrative Services'],
        "definition": "Companies that build software or provide services for hiring, employee administration, training, payroll, and benefits — covering full-cycle human-capital management."
    },
    "Humanitarian": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies whose main mission is related to giving back to the community and not for personal gain."
    },
    "Hunting": {
        "industries": ['Sports'],
        "definition": "Companies that sell firearms, ammunition, optics, apparel, or guided trips for the sport and activity of hunting wild game."
    },
    "Hydroponics": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies engaged in the science of growing plants with alternatives to soil, including sand, gravel, or liquid, in combination with added nutrients."
    },
    "ISP": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide internet access to companies, families, and mobile users. (ISP = Internet Service Provider)."
    },
    "IT Infrastructure": {
        "industries": ['Information Technology'],
        "definition": "Companies that develop a set of IT components which are the foundation of an IT service, typically physical components (computer and networking hardware and facilities), but also various software and network components."
    },
    "IT Management": {
        "industries": ['Information Technology'],
        "definition": "Companies that build software or provide services for managing enterprise IT — including asset tracking, service desk, monitoring, and IT operations."
    },
    "IaaS": {
        "industries": ['Software'],
        "definition": "Companies that offer cloud computing services in which customers rent or lease servers on-demand for computing and storage in the cloud."
    },
    "Identity and Access Management": {
        "industries": ['Information Technology', 'Privacy and Security', 'Software'],
        "definition": "Companies that build software for managing who can access what — single sign-on, multi-factor authentication, privileged access, and identity governance."
    },
    "Image Recognition": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that use computer vision to identify or detect objects or features in digital images or videos."
    },
    "Impact Investing": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that invest capital with the explicit dual aim of generating financial return and measurable positive social or environmental outcomes."
    },
    "In-Flight Entertainment": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce entertainment available to aircraft passengers during a flight or that develop technology that enables this entertainment."
    },
    "Incubators": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that help the development of new and startup companies by providing services such as management training or office space."
    },
    "Independent Music": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that produce music by musicians who are not signed to a major record label, or that provide products or services targeting these musicians."
    },
    "Indoor Positioning": {
        "industries": ['Navigation and Mapping'],
        "definition": "Companies that produce systems and devices to locate or track the precise physical position of people or objects within a building."
    },
    "Industrial": {
        "industries": ['Manufacturing'],
        "definition": "Companies that produce finished products, including manufacturing companies or any other companies that produce products."
    },
    "Industrial Design": {
        "industries": ['Design', 'Hardware'],
        "definition": "Companies that create and develop the form, features, and specifications of physical products which are intended to be mass produced."
    },
    "Industrial Engineering": {
        "industries": ['Manufacturing', 'Science and Engineering'],
        "definition": "Companies engaged in the design, improvement, and installation of integrated systems in the manufacturing and industrial processes."
    },
    "Industrial IoT": {
        "industries": ['Hardware', 'Manufacturing', 'Software'],
        "definition": "Companies that build sensors, connectivity, and analytics for industrial assets — predictive maintenance, asset performance, process monitoring, and operational technology."
    },
    "Industrial Manufacturing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that manufacture products intended for use in industrial machines OR companies that manufacture products at a large scale."
    },
    "Influencer Marketing": {
        "industries": ['Sales and Marketing', 'Advertising', 'Software'],
        "definition": "Companies that match brands with social-media creators for paid campaigns, manage talent rosters, or measure influencer-driven attribution across social platforms."
    },
    "Information Services": {
        "industries": ['Information Technology'],
        "definition": "Companies that collect, record, organize, store, preserve, retrieve, or disseminate printed or digital information or data for their clients."
    },
    "Information Technology": {
        "industries": ['Information Technology'],
        "definition": "Companies that offer products and services related to the creation, storage, processing, and management of digital data, as well as the design, implementation, and support of computer systems, networks, and software."
    },
    "Information and Communications Technology (ICT)": {
        "industries": ['Information Technology'],
        "definition": "Companies that integrate and consolidate telecom infrastructure with computing systems."
    },
    "Infrastructure": {
        "industries": ['Physical Infrastructure'],
        "definition": "Companies that finance, build, operate, or maintain large-scale physical systems — roads, bridges, power grids, water networks, and telecommunications."
    },
    "Innovation Management": {
        "industries": ['Professional Services'],
        "definition": "Companies offering products or services that provide teams with organized processes and systems to develop new ideas into finished products or services."
    },
    "InsurTech": {
        "industries": ['Financial Services'],
        "definition": "Companies that use technology innovations designed to generate savings and improve efficiency from the current insurance industry model."
    },
    "Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that provide a range of insurance policies to protect individuals and businesses against the risk of financial losses in return for regular payments of premiums."
    },
    "Intellectual Property": {
        "industries": ['Professional Services'],
        "definition": "Companies engaged in legal rights of owners of intangible creations of the human mind (e.g., music, art, designs, symbols, literature, etc.)."
    },
    "Intelligent Systems": {
        "industries": ['Artificial Intelligence', 'Data and Analytics', 'Science and Engineering'],
        "definition": "Companies that produce or develop machines with embedded internet-connected computer systems that have the capacity to gather and analyze data and communicate with other systems."
    },
    "Interior Design": {
        "industries": ['Design', 'Real Estate'],
        "definition": "Companies that provide products or services to plan, research, coordinate, or manage projects to develop or enhance the form and function of the interior of a building."
    },
    "Internal Medicine": {
        "industries": ['Health Care'],
        "definition": "Medical practices focused on the prevention, diagnosis, and management of adult diseases, including chronic and complex conditions."
    },
    "International Development": {
        "industries": ['Social Impact', 'Government and Military'],
        "definition": "Organizations that design and deliver multi-country programs to improve health, education, economic opportunity, governance, or infrastructure in developing regions."
    },
    "Internet": {
        "industries": ['Internet Services'],
        "definition": "Companies whose products or services are solely and entirely conducted online through their websites."
    },
    "Internet Radio": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that provide a digital audio service transmitted by the internet."
    },
    "Internet of Things": {
        "industries": ['Hardware', 'Internet Services', 'Software'],
        "definition": "Companies that build connected-device hardware, embedded operating systems, IoT platforms, or device-management software for consumer, industrial, or commercial IoT deployments."
    },
    "Intrusion Detection": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that help other organizations identify bugs or problems with their network device configurations."
    },
    "Janitorial Service": {
        "industries": ['Real Estate'],
        "definition": "Companies that provide residential or commercial property cleaning including professional offices, home, educational, medical, and industrial business cleaning, OR companies that develop products for use in residential or commercial cleaning."
    },
    "Jewelry": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell decorative items worn for personal adornment, including brooches, rings, necklaces, earrings, pendants, and bracelets."
    },
    "Journalism": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that work to relay an accurate account of events that occurred, OR companies that develop products or services designed to facilitate this reporting."
    },
    "Knowledge Management": {
        "industries": ['Administrative Services'],
        "definition": "Companies that build software for capturing, organizing, sharing, and retrieving institutional knowledge across teams and documents."
    },
    "LGBT": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that offer products or services designed primarily for the LGBT+ community."
    },
    "Landscaping": {
        "industries": ['Real Estate'],
        "definition": "Companies that offer products or services to make a yard or other piece of land more attractive."
    },
    "Language Learning": {
        "industries": ['Education'],
        "definition": "Companies that build products or provide services for teaching new languages — including consumer language-learning apps, online tutors, classroom curricula, and corporate-training platforms."
    },
    "Large Language Models": {
        "industries": ['Artificial Intelligence', 'Software'],
        "definition": "Companies that train, host, fine-tune, or deploy large language models — including foundation-model labs, inference platforms, and LLM-application platforms."
    },
    "Laser": {
        "industries": ['Hardware', 'Science and Engineering'],
        "definition": "Companies that produce an optical device that emits a focused beam of light, for purposes including but not limited to, barcode scanning, optical disc drive functionality, semiconductor chip manufacturing, or range measurement."
    },
    "Last Mile Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that handle the final leg of a delivery — from a local hub or fulfillment center to the end consumer — often using gig drivers or micro-fulfillment."
    },
    "Laundry and Dry-cleaning": {
        "industries": ['Clothing and Apparel'],
        "definition": "Companies that operate or service commercial and consumer laundry, dry-cleaning, garment-care, and uniform-rental businesses."
    },
    "Law Enforcement": {
        "industries": ['Government and Military', 'Privacy and Security'],
        "definition": "Companies that supply equipment, software, training, or analytics to police, sheriff, federal, and corrections agencies enforcing laws."
    },
    "Layer 2": {
        "industries": ['Blockchain and Cryptocurrency', 'Software'],
        "definition": "Companies that build scaling protocols sitting atop base blockchains — including rollups, sidechains, and other L2 networks that reduce fees and increase throughput."
    },
    "Lead Generation": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that identify and cultivate potential customers for a business's products or services."
    },
    "Lead Management": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that develop methodologies, systems, and practices designed to generate new potential customers, generally operated through a variety of marketing campaigns or programs."
    },
    "Leasing": {
        "industries": ['Financial Services'],
        "definition": "Companies whose primary service relates to renting an asset or whose primary product facilitates such rentals."
    },
    "Legal": {
        "industries": ['Professional Services'],
        "definition": "Law firms, in-house legal departments, and legal-services providers — including litigation, transactional, regulatory, and intellectual-property practices."
    },
    "Legal Tech": {
        "industries": ['Professional Services'],
        "definition": "Companies that use various technologies such as the internet, mobile devices, machine learning, or cloud computing, to compete with traditional methods in the delivery of legal services."
    },
    "Leisure": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that provide services or products for entertainment, relaxation, or recreational activities."
    },
    "Lending": {
        "industries": ['Financial Services'],
        "definition": "Companies that originate loans or build platforms that facilitate consumer, small-business, or commercial lending — including direct lenders, marketplaces, and lending infrastructure."
    },
    "Letting Agent": {
        "industries": ['Real Estate'],
        "definition": "UK-style agencies that manage rental listings, tenant placement, and lease administration on behalf of residential or commercial landlords."
    },
    "Life Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that offer a contract with an insurance company and, in exchange for premium payments, the insurance company provides a lump-sum payment known as a death benefit to beneficiaries upon the insured's death."
    },
    "Life Science": {
        "industries": ['Biotechnology', 'Science and Engineering'],
        "definition": "Companies that apply knowledge gained from the scientific study of living organisms and life processes to fields including but not limited to biotechnology, pharmaceuticals, agriculture, food processing, or genomics."
    },
    "Lifestyle": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that offer products or services designed to connote a particular stylish lifestyle (e.g., elegant modern, trendy urban, etc.), especially fashion or consumer goods companies."
    },
    "Lighting": {
        "industries": ['Hardware'],
        "definition": "Companies that are involved in the deliberate use of light to achieve practical or aesthetic effects."
    },
    "Limousine Service": {
        "industries": ['Transportation'],
        "definition": "Companies that engage in providing specialty and luxury passenger transportation services via limousine or luxury sedans, generally on a reserved basis."
    },
    "Lingerie": {
        "industries": ['Clothing and Apparel', 'Consumer Goods'],
        "definition": "Companies that design, manufacture, or sell intimate apparel — including bras, panties, sleepwear, and shapewear for retail and direct-to-consumer markets."
    },
    "Linux": {
        "industries": ['Platforms', 'Software'],
        "definition": "Companies that build, support, or sell products and services around the Linux operating system — including distributions, kernel work, and enterprise support."
    },
    "Liquid Staking": {
        "industries": ['Blockchain and Cryptocurrency', 'Financial Services'],
        "definition": "Companies that issue liquid representations of staked crypto assets, letting holders earn staking yield while keeping the asset usable in DeFi."
    },
    "Livestock": {
        "industries": ['Agriculture and Farming'],
        "definition": "Companies that raise, breed, process, or supply equipment and inputs for cattle, swine, poultry, sheep, and other farmed animals."
    },
    "Local": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that focus on customers in a particular town, city, or region — including local-services marketplaces, geo-targeted advertising, and small-business platforms serving a defined geographic area."
    },
    "Local Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that enable advertisers to advertise to audiences in a particular town, city, or region."
    },
    "Local Business": {
        "industries": ['Sales and Marketing'],
        "definition": "Independent small businesses serving a single town, city, or region — including restaurants, shops, service providers, and other community-rooted brick-and-mortar or hyper-local operators."
    },
    "Local Shopping": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that specialize in providing shopping opportunities for consumers within a localized area, OR companies that provide products or services designed for these consumers as part of their shopping."
    },
    "Location Based Services": {
        "industries": ['Data and Analytics', 'Internet Services', 'Navigation and Mapping'],
        "definition": "Companies that use real-time geo-data from a mobile device or smartphone to provide information, entertainment, or security."
    },
    "Logistics": {
        "industries": ['Transportation'],
        "definition": "Companies that plan, implement, and control the movement and storage of goods, services, or information within a supply chain and between the points of origin and consumption."
    },
    "Low-Code": {
        "industries": ['Software'],
        "definition": "Companies that build development platforms reducing the amount of hand-coded software needed for enterprise applications — typically with visual modeling and pre-built components."
    },
    "Loyalty Programs": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that develop a program run by a company that offers benefits to frequent customers."
    },
    "MMO Games": {
        "industries": ['Gaming'],
        "definition": "Companies that develop online video games that can be played by a very large number of people simultaneously."
    },
    "MOOC": {
        "industries": ['Education', 'Software'],
        "definition": "Companies that offer Massive Open Online Courses (MOOC), providing accessible, often free, online education to a large audience across the globe."
    },
    "Machine Learning": {
        "industries": ['Artificial Intelligence', 'Data and Analytics', 'Software'],
        "definition": "Companies that use artificial intelligence to develop technologies that allow computers to automatically learn and improve from training data, enabling them to make decisions without being explicitly programmed."
    },
    "Machinery Manufacturing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that design, build, and supply industrial machinery and manufacturing equipment for production lines, factories, and infrastructure."
    },
    "Made to Order": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that offer customers the chance to customize the product they want to buy."
    },
    "Management Consulting": {
        "industries": ['Professional Services'],
        "definition": "Companies that provide expert advice to organizations to help them improve their performance."
    },
    "Management Information Systems": {
        "industries": ['Information Technology'],
        "definition": "Companies that specialize in producing a computerized hardware and software information-processing system designed to support the operations of a company."
    },
    "Manufacturing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that create or produce a finished, usable product from raw materials with the help of equipment, labor, machines, or tools."
    },
    "Mapping Services": {
        "industries": ['Navigation and Mapping'],
        "definition": "Companies that make maps, including companies related to surveying or gathering data for geopositioning with the purpose of making maps of location data or information."
    },
    "MarTech": {
        "industries": ['Sales and Marketing', 'Software'],
        "definition": "Software companies that build tools for marketing teams — including campaign management, marketing automation, attribution, customer data, and content management."
    },
    "Marine Insurance": {
        "industries": ['Financial Services'],
        "definition": "Insurers and brokers specializing in coverage for ships, ocean freight, port operations, and maritime risk."
    },
    "Marine Technology": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that produce technologies for the safe use, exploitation, protection of, and intervention in the marine environment."
    },
    "Marine Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that move passengers or freight by waterway — including shipping lines, ferries, port services, and ship management."
    },
    "Market Research": {
        "industries": ['Data and Analytics', 'Design'],
        "definition": "Companies that help organizations gather information about target markets and customers to identify and analyze needs, opportunities, and competition."
    },
    "Marketing": {
        "industries": ['Sales and Marketing', 'Advertising'],
        "definition": "Companies that promote brands, products, or services on behalf of clients — including agencies, software platforms, consultancies, and integrated marketing services."
    },
    "Marketing Automation": {
        "industries": ['Sales and Marketing', 'Software'],
        "definition": "Companies that build software for orchestrating multi-channel marketing campaigns — email, web, social, SMS — with workflows, segmentation, and analytics."
    },
    "Marketplace": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that provide a physical location or an e-commerce website where multiple vendors gather to sell their products or services."
    },
    "Mechanical Contractor": {
        "industries": ['Physical Infrastructure', 'Manufacturing'],
        "definition": "Construction firms that install and service mechanical systems — HVAC, piping, plumbing, process control — in commercial, industrial, and institutional buildings."
    },
    "Mechanical Design": {
        "industries": ['Design', 'Hardware'],
        "definition": "Companies that specialize in the design of components, parts, or systems for machinery."
    },
    "Mechanical Engineering": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that design, develop, and manufacture mechanical systems and components — including engines, machinery, robotics, manufacturing equipment, and HVAC systems."
    },
    "Media and Entertainment": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce, distribute, or monetize media content across film, television, radio, music, gaming, publishing, and digital media platforms."
    },
    "Medical": {
        "industries": ['Health Care'],
        "definition": "Companies engaged in the study, diagnosis, or treatment of medical conditions — including physician practices, clinical-services firms, and medical-product manufacturers."
    },
    "Medical Device": {
        "industries": ['Health Care'],
        "definition": "Companies that produce or support the development of devices intended to be used for medical purposes, including but not limited to, the diagnosis, prevention, or treatment of a medical condition."
    },
    "Medical Spa": {
        "industries": ['Health Care'],
        "definition": "Outpatient facilities that combine licensed medical oversight with cosmetic aesthetic procedures such as injectables, lasers, body contouring, and skin treatments."
    },
    "Meeting Software": {
        "industries": ['Messaging and Telecommunications', 'Software'],
        "definition": "Companies that facilitate live conferences between two or more participants at different sites by using computer networks to transmit audio, video, and text data."
    },
    "Men's": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies focused on products, services, or content marketed primarily to men — including apparel, grooming, lifestyle, and men's-health brands."
    },
    "Men's Grooming": {
        "industries": ['Consumer Goods'],
        "definition": "Companies focused on grooming, skincare, shaving, and personal-care products marketed primarily to men."
    },
    "Mental Health": {
        "industries": ['Health Care'],
        "definition": "Companies that provide mental-health services or build technology for therapy, psychiatry, emotional-wellness coaching, and behavioral-health workforce platforms."
    },
    "Messaging": {
        "industries": ['Information Technology', 'Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that send and process email, online chat, and other direct person-to-person electronic communications."
    },
    "Micro Lending": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that lend small amounts of money at low interest to new businesses in the developing world."
    },
    "Military": {
        "industries": ['Government and Military', 'Information Technology'],
        "definition": "Companies whose primary products or services serve armed forces — including weapons systems, vehicles, electronics, training, and logistics support."
    },
    "Mineral": {
        "industries": ['Natural Resources'],
        "definition": "Companies involved with the research, mining, or processing of minerals (i.e. geological materials with a well-defined chemical composition and a specific crystal structure)."
    },
    "Mining": {
        "industries": ['Natural Resources'],
        "definition": "Companies engaged in the extraction of valuable minerals or other geological materials from the earth."
    },
    "Mining Technology": {
        "industries": ['Natural Resources'],
        "definition": "Companies that develop technology to support the extraction of valuable minerals or other geological materials from the earth."
    },
    "Mobile": {
        "industries": ['Mobile'],
        "definition": "Companies that build mobile hardware, operating systems, applications, or wireless infrastructure for smartphones, tablets, and other portable devices."
    },
    "Mobile Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that build ad-tech platforms, networks, or measurement tools specifically targeting smartphones and tablets — iOS, Android, and mobile web."
    },
    "Mobile Apps": {
        "industries": ['Apps', 'Mobile', 'Software'],
        "definition": "Companies that develop a software app, specifically for use on small, wireless computing devices (i.e. smartphones or tablets) rather than desktop or laptop computers."
    },
    "Mobile Devices": {
        "industries": ['Consumer Electronics', 'Hardware', 'Mobile'],
        "definition": "Companies that develop portable computing devices — smartphones, tablets, wearables — and the related hardware, software, and accessories."
    },
    "Mobile Payments": {
        "industries": ['Financial Services', 'Mobile', 'Payments', 'Software'],
        "definition": "Companies that provide payment services performed from or via a mobile device."
    },
    "Motion Capture": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that build hardware, software, or services for recording the movement of people or objects to drive animation, sports analysis, or biomechanics research."
    },
    "Multi-level Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies whose business model depends on a marketing strategy in which the revenue from the sale of products or services is generated from a non-salaried sales workforce, whose earnings are derived through a pyramid-shaped commission system."
    },
    "Museums and Historical Sites": {
        "industries": ['Travel and Tourism'],
        "definition": "Places that collect, preserve, interpret, and display items of artistic, cultural, or scientific significance for the education of the public."
    },
    "Music": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies broadly related to music, including the production, consumption, marketing, or promotion of it."
    },
    "Music Education": {
        "industries": ['Education', 'Media and Entertainment', 'Music and Audio'],
        "definition": "Companies focused on the field of study associated with the teaching and learning of music."
    },
    "Music Label": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that work with artists to produce and distribute music to consumers."
    },
    "Music Streaming": {
        "industries": ['Internet Services', 'Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that deliver music content continuously over the internet to a remote user, rather than requiring a user to download the content."
    },
    "Music Venues": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that own or operate live-music venues — including concert halls, clubs, festivals, amphitheaters, and venue-management software."
    },
    "Musical Instruments": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that produce an instrument created or adapted to make musical sounds."
    },
    "NFC": {
        "industries": ['Hardware'],
        "definition": "Companies that develop technologies to enable communication between two electronic devices over a distance of 4cm (1½ in) or less."
    },
    "NFT": {
        "industries": ['Blockchain and Cryptocurrency'],
        "definition": "Companies that issue, trade, or build infrastructure for non-fungible tokens — digital ownership records on blockchains used for art, collectibles, gaming, and rights management."
    },
    "NGO": {
        "industries": ['Social Impact'],
        "definition": "Non-governmental organizations operating independently of government control to deliver humanitarian, development, advocacy, or community-service programs."
    },
    "Nanotechnology": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that research or produce applications with matter on an atomic or molecular scale, common in sciences such as molecular biology, and commercial applications like nanoelectronic devices."
    },
    "National Security": {
        "industries": ['Government and Military'],
        "definition": "Companies that supply technology, analytics, or services to defense, intelligence, and homeland-security agencies for protecting national interests."
    },
    "Natural Language Processing": {
        "industries": ['Artificial Intelligence', 'Data and Analytics', 'Software'],
        "definition": "Companies that build natural-language-processing technology — programming computers to understand, generate, translate, and analyze human language for text and conversational applications."
    },
    "Natural Resources": {
        "industries": ['Natural Resources', 'Sustainability'],
        "definition": "Companies dealing with materials or substances that occur in nature and can be used for economic gain, such as minerals, forests, water, and fertile land."
    },
    "Navigation": {
        "industries": ['Navigation and Mapping'],
        "definition": "Companies that build mapping, routing, and turn-by-turn navigation products — including consumer navigation apps, automotive systems, fleet navigation, and indoor wayfinding."
    },
    "Neobank": {
        "industries": ['Financial Services'],
        "definition": "Digital-only consumer or business banks that operate without physical branches, typically built on modern banking-as-a-service infrastructure."
    },
    "Network Hardware": {
        "industries": ['Hardware'],
        "definition": "Companies that develop or support equipment that enables the use of a computer network."
    },
    "Network Security": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that work to prevent and monitor unauthorized access of or harm to a computer network."
    },
    "Neuroscience": {
        "industries": ['Biotechnology', 'Science and Engineering'],
        "definition": "Companies that conduct or commercialize research on the nervous system — including brain-computer interfaces, neuro-imaging, neurotherapeutics, and neuro-diagnostics."
    },
    "News": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that provide information about current events, whether through word of mouth, printing, postal systems, broadcasting, electronic communication, or eyewitness testimony."
    },
    "Nightclubs": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies that own or operate nightclubs, bars, and late-night entertainment venues — including dance clubs, lounges, and event-driven nightlife concepts."
    },
    "Nightlife": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Entertainment venues and bars that usually operate late into the night."
    },
    "Nintendo": {
        "industries": ['Consumer Electronics', 'Hardware', 'Platforms'],
        "definition": "Companies that develop games, accessories, or services for Nintendo platforms — including third-party publishers, peripheral makers, and Nintendo-focused content creators."
    },
    "No-Code": {
        "industries": ['Software'],
        "definition": "Companies that build visual development platforms letting non-engineers build internal tools, workflows, websites, and mobile apps without writing code."
    },
    "Non Profit": {
        "industries": ['Social Impact'],
        "definition": "Companies organized and operated for a collective, public, or social benefit and do not distribute any generated income to the company's members, directors, or officers."
    },
    "Nuclear": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that operate reactors, mine and enrich fuel, build reactor components, or develop nuclear medicine, weapons, or fusion-research technology."
    },
    "Nursing and Residential Care": {
        "industries": ['Health Care'],
        "definition": "Companies that provide or administer care to sick, elderly, or disabled individuals."
    },
    "Nutraceutical": {
        "industries": ['Health Care'],
        "definition": "Companies engaged in producing foods that contain health-giving additives and purport to have medicinal benefits."
    },
    "Nutrition": {
        "industries": ['Food and Beverage', 'Health Care'],
        "definition": "Companies that provide or obtain the nutrients necessary for health and growth."
    },
    "Observability": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build software for monitoring distributed systems — metrics, logs, traces, profiling — to detect, diagnose, and resolve production issues."
    },
    "Office Administration": {
        "industries": ['Administrative Services'],
        "definition": "Companies engaged in a set of day-to-day activities related to financial planning, record keeping and billing, personnel, physical distribution, and logistics within an organization."
    },
    "Oil and Gas": {
        "industries": ['Energy', 'Natural Resources'],
        "definition": "Companies that engage in the exploration, production, refinement, and distribution of oil and gas."
    },
    "On-Chain Analytics": {
        "industries": ['Blockchain and Cryptocurrency', 'Data and Analytics'],
        "definition": "Companies that index, analyze, and provide intelligence on blockchain transactions, smart-contract activity, wallet behavior, and on-chain market data."
    },
    "Onboarding": {
        "industries": ['Software'],
        "definition": "Companies that build software for ramping new hires — paperwork automation, role-specific training, equipment provisioning, and first-90-day workflow management."
    },
    "Online Auctions": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that facilitate virtual sales in which goods or services are virtually purchased by the highest bidder."
    },
    "Online Forums": {
        "industries": ['Community and Lifestyle', 'Internet Services'],
        "definition": "Companies that develop and run an online discussion site where people can hold conversations in the form of posted messages."
    },
    "Online Games": {
        "industries": ['Gaming'],
        "definition": "Companies that produce a video game that is primarily played through the internet."
    },
    "Online Portals": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide a webpage, allowing users an entryway to a variety of information, tools, links, and more."
    },
    "Open Banking": {
        "industries": ['Financial Services', 'Software'],
        "definition": "Companies that build APIs, aggregation, and payment-initiation services that securely connect consumer and business bank accounts to third-party applications."
    },
    "Open Source": {
        "industries": ['Software'],
        "definition": "Companies that relate especially to software for which the source code is available to the general public for use or modification."
    },
    "Operating Systems": {
        "industries": ['Platforms', 'Software'],
        "definition": "Companies that develop the system software that manages the computer's hardware and software resources, and provides common services for computer programs."
    },
    "Optical Communication": {
        "industries": ['Hardware'],
        "definition": "Companies that build fiber-optic transceivers, switches, photonic components, or systems for carrying data via light."
    },
    "Oral Care": {
        "industries": ['Consumer Goods', 'Health Care'],
        "definition": "Companies that manufacture or sell products for dental and oral hygiene — including toothpaste, mouthwash, floss, and whitening systems."
    },
    "Organic": {
        "industries": ['Sustainability'],
        "definition": "Companies that produce, certify, distribute, or retail food and consumer goods grown or made without synthetic pesticides, fertilizers, or additives."
    },
    "Organic Food": {
        "industries": ['Food and Beverage'],
        "definition": "Companies involved in the production of food without the use of chemical fertilizers, pesticides, or other artificial agents."
    },
    "Outdoor Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that specialize in advertising done outdoors, publicizing a businesses' products and services."
    },
    "Outdoors": {
        "industries": ['Sports'],
        "definition": "Companies that offer products or services related to experiencing and enjoying nature."
    },
    "Outpatient Care": {
        "industries": ['Health Care'],
        "definition": "Companies that provide medical care without admitting the patient to a hospital, including diagnosis, observation, consultation, treatment, intervention, and rehabilitation services."
    },
    "Outsourcing": {
        "industries": ['Professional Services'],
        "definition": "Companies that perform work — IT, customer service, back-office, engineering, finance — on behalf of client organizations as a managed external service."
    },
    "PC Games": {
        "industries": ['Gaming'],
        "definition": "Companies that produce games for use on a personal computer rather than a console."
    },
    "PaaS": {
        "industries": ['Software'],
        "definition": "Companies that develop a cloud computing configuration, allowing customers to develop and run applications without having to build the underlying infrastructure. (PaaS = Platform as a Service)."
    },
    "Packaging Services": {
        "industries": ['Administrative Services'],
        "definition": "Companies that support the manufacturing and logistics process by packaging products for customers, often in the food or consumer goods sectors."
    },
    "Paper Manufacturing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that produce paper, paperboard, and pulp from wood, recycled fiber, or alternative feedstocks for printing, packaging, and tissue uses."
    },
    "Parenting": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that provide products or services designed to help individuals parent their children."
    },
    "Parking": {
        "industries": ['Transportation'],
        "definition": "Companies that operate parking facilities or build software for parking management, payments, reservations, and enforcement."
    },
    "Parks": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that operate, manage, supply, or service public parks, recreation areas, and green-space facilities."
    },
    "Payments": {
        "industries": ['Financial Services', 'Payments'],
        "definition": "Companies that manage and process financial transactions between two parties from various channels, such as credit cards, e-wallets, or cash cards."
    },
    "Payroll": {
        "industries": ['Financial Services', 'Software'],
        "definition": "Companies that process employee compensation, tax withholdings, benefits deductions, and statutory filings — either as software or as a managed service."
    },
    "Peer to Peer": {
        "industries": ['Collaboration'],
        "definition": "Companies that operate within a decentralized model whereby two individuals interact to buy and sell goods and services directly with each other, without any intermediary third-party or the use of an incorporated business."
    },
    "Penetration Testing": {
        "industries": ['Information Technology', 'Privacy and Security'],
        "definition": "Companies that participate in the practice of testing a computer system, network, or web application to find security vulnerabilities that an attacker could exploit."
    },
    "Performance Management": {
        "industries": ['Software'],
        "definition": "Companies that build software for setting goals, running performance reviews, providing continuous feedback, and tying compensation to performance."
    },
    "Performing Arts": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce live presentations by actors, singers, dancers, musical groups, etc."
    },
    "Personal Branding": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that aim to create and influence positive public perception of an individual."
    },
    "Personal Care": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell consumer products used in everyday hygiene, grooming, and self-care routines — including soap, deodorant, oral care, and basic skincare."
    },
    "Personal Finance": {
        "industries": ['Health Care'],
        "definition": "Companies that help individuals or families manage their money related to budgeting, saving, investing, and spending."
    },
    "Personal Health": {
        "industries": ['Health Care'],
        "definition": "Companies that build consumer products or services for individual wellness — including fitness apps, nutrition tracking, sleep technology, and personal-health monitoring."
    },
    "Personalization": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that create and implement solutions that tailor products, services, and experiences to individual users' preferences, needs, and behaviors, often using data analytics and machine learning algorithms."
    },
    "Pet": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that build products or provide services for pet animals — including food, supplies, veterinary care, grooming, training, and pet-specific software (parent category)."
    },
    "Pet Care": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that produce or sell products and services for pet wellness, grooming, training, and daily care, excluding food and veterinary services."
    },
    "Pet Food": {
        "industries": ['Consumer Goods', 'Food and Beverage'],
        "definition": "Companies that manufacture, distribute, or sell prepared food products for domestic pets — including dry, wet, raw, and specialty-diet formulations."
    },
    "Pet Supplements": {
        "industries": ['Consumer Goods', 'Health Care'],
        "definition": "Companies that develop and sell nutritional or therapeutic supplements designed for pets — including vitamins, joint support, and digestive aids."
    },
    "Pharmaceutical": {
        "industries": ['Health Care'],
        "definition": "Companies that specialize in the research, development, manufacturing, and distribution of medications and drugs for medical use."
    },
    "Philanthropy": {
        "industries": ['Social Impact'],
        "definition": "Donor-advised funds, charitable trusts, and philanthropic advisory firms that mobilize private wealth toward social-impact causes."
    },
    "Photo Editing": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that build software or provide services for editing, retouching, and enhancing photographs — including consumer photo apps and professional retouching services."
    },
    "Photo Sharing": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that focus on the publishing or transfer of a user's digital photos online."
    },
    "Photography": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that offer a range of tools and services related to photographs, including but not limited to photography, videography, image hosting, and commercial and portrait photography."
    },
    "Physical Security": {
        "industries": ['Administrative Services', 'Privacy and Security'],
        "definition": "Companies engaged in security measures that are designed to deny unauthorized access to facilities, equipment, and resources to protect personnel and property from damage or harm."
    },
    "Plastics and Rubber Manufacturing": {
        "industries": ['Manufacturing'],
        "definition": "Companies that make goods by processing plastics materials and raw rubber."
    },
    "Platform Engineering": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build internal-developer-platform tooling — self-service infrastructure, service catalogs, golden paths, and platform-as-a-product for engineering teams."
    },
    "Playstation": {
        "industries": ['Consumer Electronics', 'Hardware', 'Platforms'],
        "definition": "Companies that develop games, accessories, or services for Sony's PlayStation platforms — including third-party publishers, peripheral makers, and PlayStation-focused content creators."
    },
    "Plumbing": {
        "industries": ['Physical Infrastructure'],
        "definition": "Companies that install and service water-supply, drainage, and gas-piping systems for residential, commercial, and industrial buildings."
    },
    "Podcast": {
        "industries": ['Media and Entertainment', 'Music and Audio'],
        "definition": "Companies that produce, host, distribute, monetize, or build tools for podcasts — including networks, recording platforms, and ad-insertion services."
    },
    "Point of Sale": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that build point-of-sale software and hardware for retailers, restaurants, and service businesses — including in-store payment terminals, integrated commerce, and SMB POS systems."
    },
    "Politics": {
        "industries": ['Government and Military'],
        "definition": "Companies that provide consulting, technology, polling, or advertising services to political campaigns, parties, advocacy groups, and government affairs teams."
    },
    "Pollution Control": {
        "industries": ['Sustainability'],
        "definition": "Companies that keep air, water, or other pollution output below specific levels."
    },
    "Ports and Harbors": {
        "industries": ['Transportation'],
        "definition": "Companies that operate or service maritime ports — including terminal operations, stevedoring, customs brokerage, and port-management software."
    },
    "Power Grid": {
        "industries": ['Energy'],
        "definition": "Companies involved with the development of a network — including transmission and distribution lines — for electricity delivery from producers to consumers."
    },
    "Precious Metals": {
        "industries": ['Natural Resources'],
        "definition": "Companies engaged in rare, naturally occurring metals with a high economic value, such as gold, silver, and platinum."
    },
    "Precision Medicine": {
        "industries": ['Health Care', 'Biotechnology'],
        "definition": "Companies that tailor clinical care to individual genetic, environmental, and lifestyle profiles — including genomics, biomarker discovery, and targeted-therapy development."
    },
    "Prediction Markets": {
        "industries": ['Financial Services'],
        "definition": "Companies engaged in exchange-traded markets created for the purpose of trading the outcome of events."
    },
    "Predictive Analytics": {
        "industries": ['Artificial Intelligence', 'Data and Analytics', 'Software'],
        "definition": "Companies that use data and statistical techniques from data mining, predictive modeling, and machine learning to analyze current and historical facts to make predictions about future or otherwise unknown events."
    },
    "Presentation Software": {
        "industries": ['Software'],
        "definition": "Companies that develop a software used to show information in the form of a slide show."
    },
    "Presentations": {
        "industries": ['Software'],
        "definition": "Companies that facilitate a speech or talk in which a new product, idea, or piece of work is shown and explained to an audience."
    },
    "Price Comparison": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that develop and maintain platforms that collect, display, and compare prices of similar products from different outlets, allowing consumers to make informed purchasing decisions."
    },
    "Primary Education": {
        "industries": ['Education'],
        "definition": "Companies that operate or supply schools, curriculum, software, or services for children in the early years of formal education."
    },
    "Printing": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that provide products or services related to producing books, newspapers, or other printed material for customers."
    },
    "Privacy": {
        "industries": ['Privacy and Security'],
        "definition": "Companies that protect a consumer's right to be free from unwanted surveillance or information disclosure and to determine whether, when, how, and to whom one's personal or organizational information is revealed."
    },
    "Private Cloud": {
        "industries": ['Hardware', 'Information Technology', 'Internet Services', 'Software'],
        "definition": "Companies that provide cloud computing resources that are not shared with other users or organizations."
    },
    "Private Social Networking": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that build platforms for users to share information exclusively with people whom they select."
    },
    "Procurement": {
        "industries": ['Transportation'],
        "definition": "Companies that build software, marketplaces, or services for sourcing suppliers, running RFPs, managing contracts, and processing purchase orders."
    },
    "Product Design": {
        "industries": ['Design'],
        "definition": "Companies that encompass the approach of building a new product from start to finish, encompassing everything from market research to product development."
    },
    "Product Management": {
        "industries": ['Software'],
        "definition": "Companies that focus on the business process of planning, developing, launching, and managing a product or service; this includes the entire lifecycle of a product, from ideation to development to go-to-market."
    },
    "Product Research": {
        "industries": ['Data and Analytics', 'Design'],
        "definition": "Companies that specialize in gathering and analyzing data on consumer needs and preferences to inform the development and improvement of products; this includes the entire lifecycle of a product, from ideation to development to go-to-market."
    },
    "Product Search": {
        "industries": ['Internet Services'],
        "definition": "Companies that build search and discovery systems for products — including site-search, marketplace-search, and shopping-comparison engines."
    },
    "Product-Led Growth": {
        "industries": ['Sales and Marketing', 'Software'],
        "definition": "Companies that build software, tooling, or consulting around product-led-growth motions — self-serve onboarding, in-product expansion, and PLG analytics."
    },
    "Productivity Tools": {
        "industries": ['Software'],
        "definition": "Companies that build software for creating documents, spreadsheets, presentations, diagrams, notes, or other knowledge work — including office suites, collaborative editors, and creator tools."
    },
    "Professional Networking": {
        "industries": ['Community and Lifestyle', 'Professional Services'],
        "definition": "Companies that provide a type of social network service, whether in-person or online, which is focused solely on interactions and relationships of a business nature rather than including personal, nonbusiness interactions."
    },
    "Professional Services": {
        "industries": ['Professional Services'],
        "definition": "Companies that offer specialized knowledge, skills, and expertise to clients, typically gained from extensive education (i.e. accountant, lawyer, engineer, doctor, architects, etc.)"
    },
    "Project Management": {
        "industries": ['Administrative Services'],
        "definition": "Companies that develop software to help with the planning and organization of a company's resources to move a specific task, event, or duty towards completion."
    },
    "PropTech": {
        "industries": ['Software', 'Real Estate', 'Data and Analytics'],
        "definition": "Software and data companies serving the real-estate industry — including listings, valuation, leasing, transaction management, building operations, and tenant experience."
    },
    "Property Development": {
        "industries": ['Real Estate'],
        "definition": "Companies that encompass activities ranging from the renovation and release of existing buildings, to the purchase of raw land and subsequent sale of developed land to others."
    },
    "Property Insurance": {
        "industries": ['Financial Services'],
        "definition": "Companies that provide insurance to protect the consumer against financial loss in the event of accidents to property such as fire, theft, and some water damage."
    },
    "Property Management": {
        "industries": ['Real Estate'],
        "definition": "Companies that provide products or services related to the operation, control, maintenance, and oversight of real estate and physical property, often acting as an intermediary between property owner and tenant."
    },
    "Psychiatry": {
        "industries": ['Health Care'],
        "definition": "Medical practices specializing in the diagnosis, treatment, and prevention of mental, emotional, and behavioral disorders."
    },
    "Psychology": {
        "industries": ['Health Care'],
        "definition": "Companies engaged in the scientific study of the human mind and its functions."
    },
    "Public Relations": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that craft and place earned-media messaging, manage corporate reputation, and run media-relations programs for brands, executives, and organizations."
    },
    "Public Safety": {
        "industries": ['Government and Military'],
        "definition": "Companies that supply technology, equipment, or services to police, fire, EMS, emergency-management, and other agencies protecting the public."
    },
    "Public Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that operate or supply public transit systems — buses, trains, subways, light rail, paratransit — typically charging fares and following fixed schedules and routes."
    },
    "Publishing": {
        "industries": ['Content and Publishing', 'Media and Entertainment'],
        "definition": "Companies that publish written content — including book publishers, magazine and newspaper publishers, digital media, and academic publishing."
    },
    "Q&A": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that provide online forums where users can ask and answer questions on a variety of topics, often relying on community-driven moderation and reputation systems to ensure quality content."
    },
    "QR Codes": {
        "industries": ['Software'],
        "definition": "Companies that develop quick response codes which are a machine-readable optical label, containing information about the item to which it is attached."
    },
    "Quality Assurance": {
        "industries": ['Professional Services'],
        "definition": "Companies that help other companies assess whether what they produce or provide is up to their standards."
    },
    "Quantified Self": {
        "industries": ['Biotechnology', 'Data and Analytics'],
        "definition": "Companies providing products and services that enable individuals to collect, analyze, and utilize their own personal data in order to improve their daily life."
    },
    "Quantum Computing": {
        "industries": ['Science and Engineering'],
        "definition": "Companies that research, develop, and manufacture quantum computing hardware, software, and related technologies, with the aim of solving complex problems that are beyond the capabilities of standard computers."
    },
    "RFID": {
        "industries": ['Hardware'],
        "definition": "Companies that develop, sell, or maintain goods utilizing a form of wireless communication that incorporates the use of electromagnetic or electrostatic coupling in the radio frequency portion of the electromagnetic spectrum."
    },
    "RISC": {
        "industries": ['Hardware'],
        "definition": "Companies that manufacture, sell, or maintain computers based on a processor designed to perform a limited set of operations extremely quickly."
    },
    "Racing": {
        "industries": ['Sports'],
        "definition": "Companies that specialize in the promotion, organization, and management of competitive motor racing events and teams."
    },
    "Railroad": {
        "industries": ['Transportation'],
        "definition": "Companies that operate or work on/with railroad tracks, rail yards, or trains."
    },
    "Reading Apps": {
        "industries": ['Apps', 'Software'],
        "definition": "Companies that build mobile or web applications for reading books, articles, or other long-form content — including e-readers, audiobook apps, and reading-focused subscription services."
    },
    "Real Estate": {
        "industries": ['Real Estate'],
        "definition": "Companies that facilitate the buying, selling, leasing, and management of residential, commercial, industrial, and/or agricultural properties, as well as companies involved in property development, property marketing, and property appraisal."
    },
    "Real Estate Investment": {
        "industries": ['Financial Services', 'Real Estate'],
        "definition": "Companies that are involved in the purchase, management, and sale or rental of real estate for profit."
    },
    "Recipes": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that curate a set of instructions for preparing a particular dish."
    },
    "Recreation": {
        "industries": ['Sports'],
        "definition": "Companies that operate leisure and recreation businesses — including tour operators, travel agencies, amusement parks, sports facilities, and outdoor-recreation services."
    },
    "Recreational Vehicles": {
        "industries": ['Transportation'],
        "definition": "Companies that produce a motor vehicle or trailer which includes living quarters designed for accommodation."
    },
    "Recruiting": {
        "industries": ['Professional Services'],
        "definition": "Companies that provide products or services that facilitate the process of identifying, sourcing, screening, shortlisting, and interviewing candidates for jobs within an organization."
    },
    "Recruiting Software": {
        "industries": ['Software'],
        "definition": "Companies that build software for recruiters — sourcing, candidate engagement, CRM, scheduling, assessment, and hiring analytics."
    },
    "Recycling": {
        "industries": ['Sustainability'],
        "definition": "Companies involved in the process of converting waste materials into new materials and objects."
    },
    "Rehabilitation": {
        "industries": ['Health Care'],
        "definition": "Companies that focus on therapy to regain or improve function after a debilitating event."
    },
    "Religion": {
        "industries": ['Community and Lifestyle'],
        "definition": "Religious institutions and the companies that supply software, content, or services to churches, synagogues, mosques, temples, and other faith communities."
    },
    "Relocation Services": {
        "industries": ['Administrative Services', 'Professional Services'],
        "definition": "Companies that coordinate employee or family relocations — including moving logistics, temporary housing, immigration, and home search."
    },
    "Remote Patient Monitoring": {
        "industries": ['Health Care', 'Software'],
        "definition": "Companies that use connected devices and software to track patient health metrics outside of clinical settings, enabling longitudinal care for chronic and post-acute conditions."
    },
    "Removals and Storage": {
        "industries": ['Administrative Services'],
        "definition": "Companies that provide household and commercial moving services — including packing, van hire, and short- or long-term self-storage."
    },
    "Renewable Energy": {
        "industries": ['Energy', 'Sustainability'],
        "definition": "Companies that develop, manufacture, finance, or operate renewable-energy systems — including solar, wind, hydro, geothermal, and biomass."
    },
    "Rental": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that operate rental businesses — including equipment rental, vehicle rental, residential and commercial property rental, and short-term rental platforms."
    },
    "Rental Property": {
        "industries": ['Real Estate'],
        "definition": "Companies that lease properties to tenants or that facilitate property leasing through products or services."
    },
    "Reputation": {
        "industries": ['Information Technology'],
        "definition": "Companies engaged in a company's overall reputation among its various stakeholders."
    },
    "Reservations": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies involved with holding someone's place, ranging from dinner reservations to parking spots."
    },
    "Residential": {
        "industries": ['Real Estate'],
        "definition": "Companies that serve the residential real-estate market — including builders, brokerages, property managers, and home-services providers."
    },
    "Resorts": {
        "industries": ['Travel and Tourism'],
        "definition": "Self-contained commercial establishments provide food, drink, lodging, sports, entertainment, and shopping on premises."
    },
    "Restaurants": {
        "industries": ['Food and Beverage'],
        "definition": "Places where people pay to sit and eat meals that are cooked and served on the premises."
    },
    "Retail": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that sell physical finished goods and products for use or consumption by individual purchasers in exchange for money."
    },
    "Retail Technology": {
        "industries": ['Commerce and Shopping', 'Hardware', 'Software'],
        "definition": "Companies that build technology for retailers — including POS, inventory, omnichannel commerce, store operations, and analytics for brick-and-mortar, e-commerce, and DTC."
    },
    "Retirement": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies focused on or related to the action of leaving one's job and ceasing to work."
    },
    "Revenue Operations": {
        "industries": ['Sales and Marketing', 'Software'],
        "definition": "Companies that build software or provide consulting to align marketing, sales, and customer-success operations under unified data, processes, and reporting."
    },
    "Reverse ETL": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build tools to sync data from warehouses back into operational SaaS applications — sales tools, marketing automation, customer success."
    },
    "Ride Sharing": {
        "industries": ['Transportation'],
        "definition": "Companies engaged in providing passengers with a private vehicle driven by its owner either for free, or for a fee, especially as arranged by means of a website or app."
    },
    "Risk Management": {
        "industries": ['Professional Services'],
        "definition": "Companies that build software, provide consulting, or supply insurance products for identifying, evaluating, and mitigating financial, operational, regulatory, or strategic risks."
    },
    "Robotics": {
        "industries": ['Hardware', 'Science and Engineering', 'Software'],
        "definition": "Companies that combine the design, construction, operation and use of robots with computer systems, for their control, sensory feedback, and information processing."
    },
    "Roku": {
        "industries": ['Consumer Electronics', 'Hardware', 'Platforms'],
        "definition": "Companies that work with or incorporate Roku (the streaming service) into their services."
    },
    "Rugby": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of rugby — including leagues, clubs, broadcasters, equipment manufacturers, and venue operators."
    },
    "SEM": {
        "industries": ['Advertising', 'Internet Services', 'Sales and Marketing'],
        "definition": "Companies that run paid-search marketing and search engine advertising — including agencies, software platforms, and bid-management tools for Google Ads, Bing Ads, and similar."
    },
    "SEO": {
        "industries": ['Internet Services', 'Sales and Marketing'],
        "definition": "Companies that work primarily to improve the quality and quantity of online traffic to a website or a web page from search engines. (SEO = Search Engine Optimization)."
    },
    "SMS": {
        "industries": ['Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that build platforms, APIs, or applications for sending, receiving, and analyzing short message service traffic at scale."
    },
    "SNS": {
        "industries": ['Software'],
        "definition": "Companies involved with an online vehicle, whose purpose is creating relationships with other people who share an interest, background, or real relationship. (SNS = Social Networking Service.)"
    },
    "STEM Education": {
        "industries": ['Education', 'Science and Engineering'],
        "definition": "Companies engaged in educating students in science, technology, engineering, and mathematics in interdisciplinary and applied approach."
    },
    "SaaS": {
        "industries": ['Software'],
        "definition": "Companies that develop a software product and sell it via a subscription over the internet rather than allowing users to download the application to their computers. (SaaS = Software as a Service)."
    },
    "Sailing": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on sailing — including boat builders, marina operators, charter services, regatta organizers, and apparel brands."
    },
    "Sales": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that provide products and services to help other businesses increase their revenue by improving their sales strategies and processes, typically by offering tools and technologies."
    },
    "Sales Automation": {
        "industries": ['Information Technology', 'Sales and Marketing', 'Software'],
        "definition": "Companies that build software to automate and streamline manual, tedious, and time-consuming tasks in the sales process including e-mail reminders, inventory control, pricing, regular documentation, standard contracts, etc."
    },
    "Sales Enablement": {
        "industries": ['Software', 'Sales and Marketing'],
        "definition": "Companies that provide content management, training delivery, and analytics for sales organizations, equipping reps with the materials and skills needed to advance deals."
    },
    "Sales Engagement": {
        "industries": ['Software', 'Sales and Marketing'],
        "definition": "Software platforms that orchestrate outbound sales activities — email sequences, dialer integration, social touchpoints — and measure response, conversion, and rep productivity."
    },
    "Sales Training": {
        "industries": ['Education', 'Sales and Marketing'],
        "definition": "Companies that deliver coaching, workshops, and curriculum to sales teams to improve prospecting, discovery, negotiation, and closing performance."
    },
    "Same Day Delivery": {
        "industries": ['Transportation'],
        "definition": "Companies that provide the service of transporting and delivering goods to customers on the same day an order is placed."
    },
    "Satellite Communication": {
        "industries": ['Hardware'],
        "definition": "Companies that produce or maintain artificial satellites that relay and amplify radio telecommunications signals."
    },
    "Scheduling": {
        "industries": ['Information Technology', 'Software'],
        "definition": "Companies that focus on the process of arranging, controlling, and optimizing work and workloads in a production process."
    },
    "Seafood": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that specialize in the farming, fishing, processing, serving, marketing, or delivery of seafood products."
    },
    "Search Engine": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide a software system that catalogs information on the internet and allows users to specify a text query that returns relevant results related to their keywords."
    },
    "Secondary Education": {
        "industries": ['Education'],
        "definition": "Companies that revolve around the stage of education which follows primary education, generally covering the early to mid–teenage years of a student's life."
    },
    "Security": {
        "industries": ['Privacy and Security'],
        "definition": "Companies that provide services to protect customers or clients from danger or threat, encompassing cybersecurity (digital security) and physical security."
    },
    "Self-Storage": {
        "industries": ['Real Estate'],
        "definition": "Companies that provide storage spaces also known as storage units which are rented to tenants usually on a short-term basis (often month-to-month)."
    },
    "Semantic Search": {
        "industries": ['Internet Services'],
        "definition": "Companies that build search systems using natural-language understanding and context to return results matching user intent rather than literal keyword matches."
    },
    "Semantic Web": {
        "industries": ['Internet Services'],
        "definition": "Companies engaged in the file format genre semantic web that has the ultimate goal of making a machine understand internet data."
    },
    "Semiconductor": {
        "industries": ['Hardware', 'Science and Engineering'],
        "definition": "Companies that design or manufacture integrated circuits, memory, microcontrollers, and the equipment, materials, and EDA tools used to produce them."
    },
    "Senior Care": {
        "industries": ['Health Care'],
        "definition": "Companies that provide care, housing, or services to older adults — including home care, assisted living, hospice, memory care, and senior-focused tech."
    },
    "Sensor": {
        "industries": ['Hardware'],
        "definition": "Companies that develop devices whose purpose is to detect events or changes in its environment and send the information to other electronics."
    },
    "Sharing Economy": {
        "industries": ['Collaboration'],
        "definition": "Companies that allow individuals to lend goods or offer services to other individuals at rates cheaper than those offered through traditional entities."
    },
    "Shipping": {
        "industries": ['Transportation'],
        "definition": "Companies that move goods and cargo by land, sea, or air — including ocean carriers, freight forwarders, parcel carriers, and shipping software."
    },
    "Shipping Broker": {
        "industries": ['Transportation'],
        "definition": "Companies that act as intermediaries arranging the purchase, sale, or chartering of ships and cargo space between owners and shippers."
    },
    "Shoes": {
        "industries": ['Clothing and Apparel', 'Consumer Goods'],
        "definition": "Companies that design, manufacture, or sell footwear — including athletic, casual, dress, and specialty shoes across direct-to-consumer, wholesale, and retail channels."
    },
    "Shopping": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that operate or supply retail commerce — including stores, malls, e-commerce platforms, shopping intelligence, mystery-shopping, and customer-experience services."
    },
    "Shopping Mall": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that develop, own, or operate shopping centers — including indoor malls, lifestyle centers, and outlet complexes."
    },
    "Simulation": {
        "industries": ['Software'],
        "definition": "Companies that create and utilize computer-based models to mimic real-world processes, systems, and environments."
    },
    "Site Reliability Engineering": {
        "industries": ['Software', 'Information Technology'],
        "definition": "Companies that build software or services for operating production systems reliably at scale — including incident response, SLOs, error budgets, and chaos engineering."
    },
    "Skiing": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on skiing and snow sports — including resort operators, equipment manufacturers, apparel brands, and instruction services."
    },
    "Skill Assessment": {
        "industries": ['Education'],
        "definition": "Companies that provide tests designed to help individuals or employers evaluate an individual's ability to perform particular tasks."
    },
    "Skin Care": {
        "industries": ['Consumer Goods', 'Health Care'],
        "definition": "Companies that develop and sell topical products for cleansing, moisturizing, treating, or protecting skin — including serums, creams, sunscreens, and acne treatments."
    },
    "Smart Building": {
        "industries": ['Real Estate'],
        "definition": "Companies that use Internet of Things devices combined with building management systems to monitor, analyze, and manage a building's operations, involving the installation and use of advanced and integrated building technology systems."
    },
    "Smart Cities": {
        "industries": ['Real Estate'],
        "definition": "Companies that supply sensors, software, or analytics for monitoring and managing urban operations — transportation, energy, utilities, public safety, and infrastructure — to improve efficiency and quality of life."
    },
    "Smart Contracts": {
        "industries": ['Blockchain and Cryptocurrency', 'Software'],
        "definition": "Companies that develop self-executing on-chain programs, smart-contract languages, security tooling, or smart-contract auditing services."
    },
    "Smart Home": {
        "industries": ['Consumer Electronics', 'Real Estate'],
        "definition": "Companies producing and selling products and services that make homes more automated, connected, and convenient."
    },
    "Snack Food": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that produce, package, or sell shelf-stable snack products — including chips, crackers, bars, jerky, and other portion-sized eating occasions."
    },
    "Soccer": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of soccer — including leagues, clubs, broadcasters, equipment manufacturers, and venue operators."
    },
    "Social": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies whose mission centers on addressing a social problem — including mission-driven businesses and social-enterprise platforms."
    },
    "Social Assistance": {
        "industries": ['Government and Military'],
        "definition": "Companies and nonprofits that provide services to individuals and families in need — including food banks, housing assistance, family services, and case-management software."
    },
    "Social Bookmarking": {
        "industries": ['Content and Publishing'],
        "definition": "Companies that offer users the ability to add, annotate, edit, and share bookmarks of web documents."
    },
    "Social CRM": {
        "industries": ['Information Technology', 'Sales and Marketing', 'Software'],
        "definition": "Companies that provide software or services that enable businesses to engage with their customers through social media channels, integrating these interactions with traditional customer relationship management (CRM) tools."
    },
    "Social Entrepreneurship": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that pursue commercial models explicitly designed to generate measurable social or environmental impact alongside financial returns."
    },
    "Social Impact": {
        "industries": ['Social Impact'],
        "definition": "Companies that intend to aid in or positively impact social injustices or challenges (i.e. homelessness, racial inequality, climate change, etc.)."
    },
    "Social Media": {
        "industries": ['Internet Services', 'Media and Entertainment'],
        "definition": "Companies that offer products or services designed to manage or enhance that creation or sharing of information."
    },
    "Social Media Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that build software or provide services for advertising on social platforms — including campaign management, creative production, and analytics for Meta, TikTok, LinkedIn, and X."
    },
    "Social Media Management": {
        "industries": ['Internet Services', 'Sales and Marketing'],
        "definition": "Companies engaged in the process of creating, scheduling, analyzing, and engaging with content posted on social media platforms."
    },
    "Social Media Marketing": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies that offer products or services to promote brands, products, or services using social media platforms."
    },
    "Social Network": {
        "industries": ['Internet Services'],
        "definition": "Companies that develop a website, application, or platform that enables users to communicate with each other by virtually creating and sharing information; this excludes companies that offer social media products or services but are not social networks."
    },
    "Social News": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that feature user-posted stories which are ranked based on popularity."
    },
    "Social Recruiting": {
        "industries": ['Professional Services'],
        "definition": "Companies that build software or services for sourcing, engaging, and hiring candidates through social-media platforms, forums, and online communities."
    },
    "Social Shopping": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies engaged in social networking services to share their latest purchases, wants, deals, etc."
    },
    "Software": {
        "industries": ['Software'],
        "definition": "Companies whose primary product is a computer application or whose services facilitate the development of computer applications."
    },
    "Software Engineering": {
        "industries": ['Science and Engineering', 'Software'],
        "definition": "Companies whose primary product or service deals with the design, development, testing, and maintenance of software applications."
    },
    "Solar Energy": {
        "industries": ['Energy', 'Natural Resources', 'Sustainability'],
        "definition": "Companies that manufacture solar panels and components, develop and operate solar projects, or build software and financing for residential, commercial, and utility solar."
    },
    "Space Travel": {
        "industries": ['Transportation'],
        "definition": "Companies engaged in space exploration and transport — including launch vehicles, satellites, suborbital tourism, and crewed-space missions."
    },
    "Spam Filtering": {
        "industries": ['Information Technology'],
        "definition": "Companies that provide spam filtering products and services to identify and block unsolicited spam in various forms (i.e spam texts, spam emails, spam calls, website bot spam, and more)."
    },
    "Speech Recognition": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that develop, improve, sell, or maintain technology to recognize voices."
    },
    "Sponsorship": {
        "industries": ['Sales and Marketing'],
        "definition": "Companies engaged in a form of advertising in which companies pay to be associated with certain events."
    },
    "Sporting Goods": {
        "industries": ['Commerce and Shopping', 'Sports'],
        "definition": "Companies that create or deal with sports equipment sold as a commodity."
    },
    "Sports": {
        "industries": ['Sports'],
        "definition": "Companies that are involved in producing, facilitating, promoting, or organizing any activity, experience, or business enterprise focused on sports."
    },
    "Stablecoin": {
        "industries": ['Blockchain and Cryptocurrency', 'Payments', 'Financial Services'],
        "definition": "Companies that issue, custody, or build payment rails for fiat-pegged crypto tokens — including USD-, EUR-, and asset-backed stablecoins."
    },
    "Staffing Agency": {
        "industries": ['Administrative Services'],
        "definition": "Companies that have employees who can be hired out for temporary or long-term work."
    },
    "Stock Exchanges": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that provide, support, or are dedicated to a centralized location where the shares of publicly traded companies are bought and sold."
    },
    "Subscription Commerce": {
        "industries": ['Commerce and Shopping', 'Software'],
        "definition": "Companies that operate subscription-based product or service businesses — including consumer subscription boxes — and the software platforms enabling recurring commerce."
    },
    "Supply Chain Management": {
        "industries": ['Transportation', 'Software', 'Information Technology'],
        "definition": "Companies that manage the flow of goods and services, involving the movement and storage of raw materials, of work-in-process inventory, and of finished goods from point-of-origin to point-of-consumption."
    },
    "Surfing": {
        "industries": ['Sports'],
        "definition": "Companies involved in all aspects of the sport of surfing, from training to the manufacturing of surfing-related goods."
    },
    "Sustainability": {
        "industries": ['Sustainability'],
        "definition": "Companies that develop and promote environmentally friendly products, services, and practices that minimize negative impacts on the environment and natural resources, while promoting social and economic well-being."
    },
    "Swimming": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport and activity of swimming — including pool operators, swim apparel and equipment makers, training programs, and competitive swimming organizations."
    },
    "TV": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that broadcast programs on, develop products for, or otherwise primarily focus on television."
    },
    "TV Production": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that work to create television programs, including companies involved in development, pre-production, filming, post-production, and distribution."
    },
    "Table Tennis": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on table tennis — including equipment manufacturers, leagues, broadcasters, and venue operators."
    },
    "Talent Acquisition": {
        "industries": ['Software', 'Professional Services'],
        "definition": "Companies that build software or provide services for sourcing, evaluating, and hiring candidates — including sourcing tools, assessment platforms, and recruitment-marketing software."
    },
    "Task Management": {
        "industries": ['Software'],
        "definition": "Companies that provide products or services to help individuals or teams manage all aspects of a task through its life cycle."
    },
    "Taxi Service": {
        "industries": ['Transportation'],
        "definition": "Companies that are involved in the transportation of persons in a taxi cab (or any motor vehicle equipped or designed to carry 12 passengers or less) for a non-shared ride, based on time and distance traveled as measured by a taximeter."
    },
    "Tea": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that source, blend, manufacture, distribute, or retail tea — including specialty tea brands, ready-to-drink tea beverages, and tea-shop chains."
    },
    "Technical Support": {
        "industries": ['Information Technology'],
        "definition": "Companies that offer assistance and troubleshooting services to clients experiencing issues with technological products such as software, hardware, networking equipment, mobile phones, printers, etc."
    },
    "Teenagers": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that develop or sell products and services targeted primarily at teenagers — including teen-focused apparel, entertainment, beauty, social, and lifestyle brands."
    },
    "Telecommunications": {
        "industries": ['Hardware'],
        "definition": "Companies that transmit information over wire, radio, optical, or other electromagnetic systems."
    },
    "Telehealth": {
        "industries": ['Health Care', 'Software'],
        "definition": "Broad-category companies that use telecommunications technology to deliver health-care services, education, or administration remotely."
    },
    "Telemedicine": {
        "industries": ['Health Care', 'Software'],
        "definition": "Companies that deliver clinical care remotely via video, asynchronous messaging, or phone — including primary care, urgent care, mental health, and specialty telemedicine platforms."
    },
    "Tennis": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on the sport of tennis — including racket and apparel manufacturers, tour operators, clubs, broadcasters, and venue operators."
    },
    "Test and Measurement": {
        "industries": ['Data and Analytics'],
        "definition": "Companies that offer assessment instruments or services to other organizations."
    },
    "Text Analytics": {
        "industries": ['Data and Analytics', 'Software'],
        "definition": "Companies that build software for extracting structured insight from unstructured text — including sentiment analysis, entity extraction, topic modeling, and document understanding."
    },
    "Textbook": {
        "industries": ['Education'],
        "definition": "Companies engaged in books used as a standard work for the study of a particular subject."
    },
    "Textiles": {
        "industries": ['Manufacturing'],
        "definition": "Companies that produce, distribute, or finish cloth and woven fabrics — including yarn, weaving, dyeing, technical textiles, and finished-textile products."
    },
    "Theatre": {
        "industries": ['Media and Entertainment'],
        "definition": "Companies that produce, stage, license, or supply theatrical performances — including production companies, venues, ticketing platforms, and equipment suppliers."
    },
    "Therapeutics": {
        "industries": ['Health Care'],
        "definition": "Companies that develop, manufacture, or commercialize drugs, biologics, devices, or digital therapies that treat disease."
    },
    "Ticketing": {
        "industries": ['Events', 'Media and Entertainment'],
        "definition": "Companies engaged in the production, selling, or management of tickets."
    },
    "Timber": {
        "industries": ['Natural Resources'],
        "definition": "Companies that work with employees, equipment, machinery, space, property, management, and technology created and operating to provide timber and timber-related products to a wide variety of customers."
    },
    "Timeshare": {
        "industries": ['Real Estate', 'Travel and Tourism'],
        "definition": "Companies that develop, operate, sell, or resell shared-ownership vacation properties — including timeshare resorts, fractional ownership, and timeshare exchange networks."
    },
    "Tizen": {
        "industries": ['Platforms'],
        "definition": "Companies engaged in a Linux-based mobile operating system backed by the Linux Foundation but developed and used primarily by Samsung Electronics."
    },
    "Tobacco": {
        "industries": ['Consumer Goods', 'Food and Beverage'],
        "definition": "Companies engaged in tobacco growing, processing, distribution, retail, or alternative-product development — including cigarettes, cigars, vaping, and reduced-risk nicotine products."
    },
    "Token Issuer": {
        "industries": ['Blockchain and Cryptocurrency'],
        "definition": "Foundations, labs entities, and DAOs that have launched a native cryptocurrency or governance token tied to a protocol, application, or community."
    },
    "Tokenization": {
        "industries": ['Blockchain and Cryptocurrency', 'Financial Services'],
        "definition": "Companies that represent real-world assets — real estate, equities, debt, commodities — as blockchain tokens for trading, settlement, or programmable ownership."
    },
    "Tour Operator": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that organize and sell guided travel experiences — including group tours, expedition trips, specialty tours, and tour-booking marketplaces."
    },
    "Tourism": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that attract, host, transport, or entertain travelers — including tour operators, hotels, attractions, destination marketing, and travel agencies."
    },
    "Toys": {
        "industries": ['Consumer Goods'],
        "definition": "Companies that design, manufacture, or sell toys, games, dolls, plush, building sets, and adjacent play-and-learning products for children and collectors."
    },
    "Trade Shows": {
        "industries": ['Administrative Services'],
        "definition": "Companies engaged in or are exhibitions at which businesses in a particular industry promote their products and services."
    },
    "Trading Platform": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that develop software enabling users to monitor investment accounts and to place trades for stocks, bonds, currencies, commodities, derivatives, and other financial products through financial intermediaries."
    },
    "Training": {
        "industries": ['Education'],
        "definition": "Companies that teach individuals specific skills needed for a particular job or task."
    },
    "Transaction Processing": {
        "industries": ['Financial Services', 'Payments', 'Software'],
        "definition": "Companies that focus on information processing that is divided into individual, indivisible operations called transactions."
    },
    "Translation Service": {
        "industries": ['Professional Services'],
        "definition": "Companies that translate text or speech between languages — including human translation agencies, machine-translation platforms, and localization services."
    },
    "Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that move people, goods, or animals between locations — including operators, infrastructure, logistics software, and mobility services."
    },
    "Travel": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that arrange, book, or supply services for personal or business travel — including booking platforms, agencies, tour operators, and corporate travel managers."
    },
    "Travel Accommodations": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that provide lodging — including hotels, vacation rentals, hostels, serviced apartments, and short-term-rental platforms."
    },
    "Travel Agency": {
        "industries": ['Travel and Tourism'],
        "definition": "Companies that book, plan, or sell travel arrangements on behalf of travelers — including online travel agencies, corporate travel managers, and traditional retail agencies."
    },
    "Treasury Management": {
        "industries": ['Financial Services', 'Software'],
        "definition": "Companies that build software or services for cash management, liquidity forecasting, bank-account reconciliation, FX hedging, and payment orchestration for corporate finance teams."
    },
    "Tutoring": {
        "industries": ['Education'],
        "definition": "Companies that offer private academic support to people who are in need of educational assistance."
    },
    "Twitter": {
        "industries": ['Platforms'],
        "definition": "Companies whose primary product is built on the Twitter/X platform — including Twitter-data analytics, social-listening tools, and X-API integrations."
    },
    "UX Design": {
        "industries": ['Design'],
        "definition": "Companies that focus on designing the user experience of interacting with a product or website. This includes all aspects of a user's interaction with the product or website, i.e. usability, accessibility, desirability, and performance."
    },
    "Ultimate Frisbee": {
        "industries": ['Sports'],
        "definition": "Companies that widely relate to Ultimate Frisbee, a sport in which players seek to score points by passing a frisbee to a teammate over the opposing team's goal line. This includes Ultimate Frisbee leagues, teams, media outlets, and official equipment manufacturers."
    },
    "Unified Communications": {
        "industries": ['Information Technology', 'Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that provide software or services that integrate various communication methods within a business, such as voice, video, messaging, and collaboration tools, into a single, cohesive system."
    },
    "Universities": {
        "industries": ['Education'],
        "definition": "Companies that operate degree-granting institutions or supply software, content, and services to colleges and universities."
    },
    "Usability Testing": {
        "industries": ['Data and Analytics', 'Design'],
        "definition": "Companies that specialize in evaluating and providing feedback on the user experience of websites, applications, or products to improve their ease of use, efficiency, and satisfaction for end-users."
    },
    "Vacation Rental": {
        "industries": ['Real Estate', 'Travel and Tourism'],
        "definition": "Companies engaged in the reservations of lodging for short-term periods."
    },
    "Vending and Concessions": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that operate vending machines, micro-markets, or concessions at venues — including office snacks, stadium concessions, and unattended retail."
    },
    "Venture Capital": {
        "industries": ['Financial Services', 'Lending and Investments'],
        "definition": "Companies that invest or facilitate investments in early-stage, high-potential, and innovative startup companies, often in exchange for equity ownership in the company and a goal of generating significant returns on investment."
    },
    "Vertical Search": {
        "industries": ['Internet Services'],
        "definition": "Companies involved with search engines that focus on a specific domain, or vertical."
    },
    "Veterinary": {
        "industries": ['Health Care'],
        "definition": "Companies that provide veterinary medical services, sell veterinary pharmaceuticals and supplies, or build software for animal-health practices."
    },
    "Video": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that focus primarily on the production, editing, distribution, hosting, and/or broadcasting of video content."
    },
    "Video Advertising": {
        "industries": ['Advertising', 'Sales and Marketing'],
        "definition": "Companies that build platforms or supply services for video advertising — including CTV/OTT ad tech, in-stream video ads, programmatic video, and video creative."
    },
    "Video Chat": {
        "industries": ['Information Technology', 'Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that build software or hardware for one-to-one or small-group video calling over the internet — including consumer video-call apps and embedded video-chat APIs."
    },
    "Video Conferencing": {
        "industries": ['Hardware', 'Information Technology', 'Internet Services', 'Messaging and Telecommunications', 'Software'],
        "definition": "Companies that build software or hardware for multi-party video meetings — including enterprise meeting platforms, conference-room systems, webinars, and meeting-AI tools."
    },
    "Video Editing": {
        "industries": ['Content and Publishing', 'Media and Entertainment', 'Video'],
        "definition": "Companies that build software or provide services for editing video content — including post-production tools, AI video editors, and freelance video-editing services."
    },
    "Video Games": {
        "industries": ['Gaming'],
        "definition": "Companies that develop or publish video games — across PC, console, mobile, and cloud-gaming platforms — including studios, publishers, and live-service operators."
    },
    "Video Streaming": {
        "industries": ['Content and Publishing', 'Media and Entertainment', 'Video'],
        "definition": "Companies that deliver video content continuously over the internet to a remote user."
    },
    "Video on Demand": {
        "industries": ['Media and Entertainment', 'Video'],
        "definition": "Companies that deliver or support the delivery of movies, programs, or sports events to a TV when the customer requests it."
    },
    "Virtual Assistant": {
        "industries": ['Software'],
        "definition": "Companies that provide remote administrative, executive, or specialized support staff to businesses — typically on a fractional or contract basis."
    },
    "Virtual Currency": {
        "industries": ['Financial Services', 'Payments', 'Software'],
        "definition": "Companies that produce and deal with electronic representations of monetary value that may be issued, managed, and controlled."
    },
    "Virtual Desktop": {
        "industries": ['Software'],
        "definition": "Companies engaged in computer operating systems that run virtually, as opposed to directly on the user's endpoint hardware."
    },
    "Virtual Goods": {
        "industries": ['Commerce and Shopping', 'Software'],
        "definition": "Companies that produce or sell digital items used inside applications and games — including in-game items, virtual currency, NFT goods, and metaverse assets."
    },
    "Virtual Reality": {
        "industries": ['Hardware', 'Software'],
        "definition": "Companies involved in creating computer-generated simulations of a 3D image or environment that can be interacted with in a seemingly real or physical way."
    },
    "Virtual Workforce": {
        "industries": ['Administrative Services'],
        "definition": "Companies that offer services or technology that enables businesses to employ and manage remote workers across various locations."
    },
    "Virtual World": {
        "industries": ['Community and Lifestyle', 'Media and Entertainment', 'Software'],
        "definition": "Companies involved in computer-based simulated environments, many of which support user-created personal avatars."
    },
    "Virtualization": {
        "industries": ['Hardware', 'Information Technology', 'Software'],
        "definition": "Companies that develop software or hardware solutions that allow multiple operating systems and applications to run on a single physical server or computer by creating a virtual (rather than physical) version of the device or resource."
    },
    "Visual Search": {
        "industries": ['Internet Services'],
        "definition": "Companies that specialize in the use of real world images (screenshots, Internet images, or photographs) as the stimuli for online searches."
    },
    "VoIP": {
        "industries": ['Information Technology', 'Internet Services', 'Messaging and Telecommunications'],
        "definition": "Companies that develop VoIP services for residential and commercial customers. (VoIP = Voiceover Internet Protocol)."
    },
    "Vocational Education": {
        "industries": ['Education'],
        "definition": "Companies that prepare people for a skilled craft as an artisan, trade as a tradesperson, or work as a technician."
    },
    "Volleyball": {
        "industries": ['Sports'],
        "definition": "Companies whose business centers on volleyball — including leagues, equipment manufacturers, apparel brands, and venue operators."
    },
    "Vulnerability Management": {
        "industries": ['Privacy and Security', 'Software'],
        "definition": "Companies that scan systems for security weaknesses, prioritize remediation, and track exposure across endpoints, cloud, applications, and networks."
    },
    "Warehousing": {
        "industries": ['Transportation'],
        "definition": "Companies that temporarily store goods before they are later sold or distributed."
    },
    "Waste Management": {
        "industries": ['Sustainability'],
        "definition": "Companies that process and manage waste, i.e. garbage disposal, refuse disposal, recycling, etc."
    },
    "Water": {
        "industries": ['Natural Resources'],
        "definition": "Companies that produce, treat, distribute, manage, or develop technology for fresh, waste, or process water across municipal, industrial, and consumer uses."
    },
    "Water Purification": {
        "industries": ['Sustainability'],
        "definition": "Companies engaged in the process of removing undesirable chemicals, biological contaminants, suspended solids, and gasses from water."
    },
    "Water Transportation": {
        "industries": ['Transportation'],
        "definition": "Companies that move passengers or goods over water — including ferries, cruise lines, charter boats, and short-sea shipping operators."
    },
    "Wealth Management": {
        "industries": ['Financial Services'],
        "definition": "Companies that work with clients to maximize their overall financial situation through asset management and various forms of broader financial planning."
    },
    "Wearables": {
        "industries": ['Consumer Electronics', 'Hardware'],
        "definition": "Companies that produce computing devices intended to be worn on the body."
    },
    "Web Apps": {
        "industries": ['Apps', 'Software'],
        "definition": "Companies that build software applications delivered through web browsers — including SaaS web apps, web-based productivity tools, and progressive web apps."
    },
    "Web Browsers": {
        "industries": ['Internet Services', 'Software'],
        "definition": "Companies that develop web browsers — including major consumer browsers, specialty browsers (privacy-focused, productivity-focused), and browser-based enterprise platforms."
    },
    "Web Design": {
        "industries": ['Design'],
        "definition": "Companies that support the design and user experience aspect of website development."
    },
    "Web Development": {
        "industries": ['Software'],
        "definition": "Companies that either build websites for clients or provide products/services to enable customers to build their own."
    },
    "Web Hosting": {
        "industries": ['Internet Services'],
        "definition": "Companies that provide the technologies (e.g., servers) and services (e.g., DNS and other configurations) required to maintain a website and make it accessible online."
    },
    "Web3": {
        "industries": ['Blockchain and Cryptocurrency', 'Software'],
        "definition": "Companies building decentralized internet applications and infrastructure on blockchains — including wallets, identity, social, gaming, and developer platforms."
    },
    "Web3 Investor": {
        "industries": ['Blockchain and Cryptocurrency', 'Financial Services', 'Lending and Investments'],
        "definition": "Investment firms, venture capital funds, and hedge funds that focus specifically on blockchain, cryptocurrency, DeFi, NFTs, and other Web3 technologies. Includes crypto VCs, token funds, and Web3-focused angel investors."
    },
    "WebOS": {
        "industries": ['Platforms'],
        "definition": "Companies that build, run, or supply applications for WebOS — an LG-owned, Linux-based operating system used in smart TVs and embedded devices."
    },
    "Wedding": {
        "industries": ['Community and Lifestyle', 'Events'],
        "definition": "Companies that provide products or services for weddings — including venues, planners, photographers, dress designers, florists, and wedding-specific marketplaces."
    },
    "Wellness": {
        "industries": ['Health Care'],
        "definition": "Companies that enable consumers to incorporate wellness activities and lifestyles into their daily lives."
    },
    "Wholesale": {
        "industries": ['Commerce and Shopping'],
        "definition": "Companies that sell and purchase great quantities of products from manufacturers, farmers and other producers, or vendors."
    },
    "Wind Energy": {
        "industries": ['Energy', 'Natural Resources', 'Sustainability'],
        "definition": "Companies engaged in wind-power generation — including turbine manufacturers, project developers, operators, and wind-farm software providers."
    },
    "Windows": {
        "industries": ['Platforms'],
        "definition": "Companies that build software for, around, or related to Microsoft operating systems."
    },
    "Windows Phone": {
        "industries": ['Consumer Electronics', 'Hardware', 'Mobile', 'Platforms'],
        "definition": "Companies that work with or incorporate Microsoft's Windows Phone into their services."
    },
    "Wine And Spirits": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that produce, distribute, or retail wine, distilled spirits, and related alcoholic beverages excluding beer."
    },
    "Winery": {
        "industries": ['Food and Beverage'],
        "definition": "Companies that house a building or property which produces wine or a business involved in the production of wine such as a wine company."
    },
    "Wired Telecommunications": {
        "industries": ['Messaging and Telecommunications'],
        "definition": "Companies that produce telecommunication systems that are not wireless."
    },
    "Wireless": {
        "industries": ['Hardware', 'Mobile'],
        "definition": "Companies that facilitate the transfer of information using remote communication technologies (i.e., two-way radios or cell phones or using cordless devices (i.e. keyboards, headsets, garage door openers, etc.)."
    },
    "Women's": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies focused on products, services, or content marketed primarily to women — including apparel, beauty, lifestyle, and women's-health brands."
    },
    "Wood Processing": {
        "industries": ['Manufacturing'],
        "definition": "Companies involved in the transformation of turning trees into paper, lumber, or any other wood product."
    },
    "Workforce Management": {
        "industries": ['Software', 'Professional Services'],
        "definition": "Software platforms that handle scheduling, time tracking, labor forecasting, and compliance for organizations managing hourly, frontline, or distributed teams."
    },
    "Xbox": {
        "industries": ['Consumer Electronics', 'Hardware', 'Platforms'],
        "definition": "Companies that specifically support Microsoft's Xbox and its services, i.e. building, game development, branding, etc."
    },
    "Young Adults": {
        "industries": ['Community and Lifestyle'],
        "definition": "Companies that develop or sell products and services targeted at individuals in their teens and early twenties — including young-adult apparel, entertainment, and lifestyle brands."
    },
    "Zero Trust": {
        "industries": ['Privacy and Security'],
        "definition": "Companies that build security architectures and products based on never-trust-always-verify principles — granular access control, continuous authentication, and microsegmentation."
    },
    "eSports": {
        "industries": ['Sports'],
        "definition": "Companies that organize, broadcast, sponsor, or support competitive multiplayer video gaming — including tournaments, teams, streaming platforms, and team management."
    },
    "iOS": {
        "industries": ['Mobile', 'Platforms', 'Software'],
        "definition": "Companies that develop software for, based off of, and work exclusively with Apple's proprietary mobile operating system."
    },
    "mHealth": {
        "industries": ['Health Care', 'Mobile'],
        "definition": "Companies that use mobile phones and other wireless technology in medical care. (mHealth = mobile Health)."
    },
    "macOS": {
        "industries": ['Platforms', 'Software'],
        "definition": "Companies that work with and build products for/off of a series of proprietary graphical operating systems developed and marketed by Apple since 2001."
    },
}
