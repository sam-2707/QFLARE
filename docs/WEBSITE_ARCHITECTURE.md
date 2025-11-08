# QFLARE Website Architecture - Mermaid Diagrams

## 1. Complete Website Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser<br/>Chrome/Firefox/Safari]
        Browser --> React[React Application<br/>Port 3000]
    end
    
    subgraph "Frontend - React SPA"
        React --> Router[React Router]
        
        Router --> Login[Login Page<br/>/login]
        Router --> Dashboard[Main Dashboard<br/>/dashboard]
        Router --> Privacy[Privacy Dashboard<br/>/privacy]
        Router --> Security[Security Dashboard<br/>/security]
        Router --> Training[Training Dashboard<br/>/training]
        Router --> Admin[Admin Panel<br/>/admin]
        
        Dashboard --> Components[Reusable Components]
        Components --> Cards[Summary Cards]
        Components --> Charts[Traffic Charts]
        Components --> Tables[Device Lists]
        Components --> Alerts[Alert Feed]
        Components --> Notif[Notification Bell]
    end
    
    subgraph "State Management"
        Components --> Redux[Redux Store]
        Redux --> AuthSlice[Auth State]
        Redux --> DataSlice[Dashboard Data]
        Redux --> UISlice[UI State]
    end
    
    subgraph "API Layer"
        Components --> API[API Service<br/>utils/api.ts]
        API --> Auth[Authentication<br/>JWT Tokens]
        API --> Timeout[Request Timeout<br/>5-20s]
        API --> Retry[Retry Logic<br/>Exponential Backoff]
    end
    
    subgraph "Backend - FastAPI"
        API --> FastAPI[FastAPI Server<br/>Port 8000]
        
        FastAPI --> AuthAPI[/api/auth/*<br/>Login/Register]
        FastAPI --> PrivacyAPI[/api/privacy/*<br/>Metrics/Budget]
        FastAPI --> SecurityAPI[/api/security/*<br/>Byzantine/Audit]
        FastAPI --> TrainingAPI[/api/training/*<br/>Start/Stop/Status]
        FastAPI --> NotifAPI[/api/notifications<br/>Alerts]
        FastAPI --> AdminAPI[/api/admin/*<br/>User Management]
    end
    
    subgraph "Database Layer"
        AuthAPI --> DB[(PostgreSQL<br/>Users & Sessions)]
        TrainingAPI --> DB
        AdminAPI --> DB
        
        PrivacyAPI --> Redis[(Redis<br/>Real-time Data)]
        SecurityAPI --> Redis
        NotifAPI --> Redis
    end
    
    subgraph "ML/Training Layer"
        TrainingAPI --> FL[Federated Learning<br/>Coordinator]
        FL --> Edge1[Edge Node 1]
        FL --> Edge2[Edge Node 2]
        FL --> EdgeN[Edge Node N]
    end
    
    style Browser fill:#e3f2fd
    style React fill:#e1f5ff
    style Dashboard fill:#fff3e0
    style Privacy fill:#f3e5f5
    style Security fill:#ffebee
    style Training fill:#e8f5e9
    style Admin fill:#fce4ec
    style FastAPI fill:#fff3e0
    style DB fill:#e8f5e9
    style Redis fill:#ffebee
```

## 2. Dashboard Component Architecture (Based on NIDS Control Center)

```mermaid
graph TB
    subgraph "Main Dashboard Layout"
        Header[Header Bar<br/>Logo + User + Notifications]
        
        Header --> TopRow[Top Statistics Row]
        TopRow --> Card1[Packet Loss<br/>111]
        TopRow --> Card2[Network Traffic<br/>23.4 KB]
        TopRow --> Card3[Bandwidth<br/>4]
        TopRow --> Card4[Stream Priority<br/>0]
        TopRow --> Card5[Others<br/>0]
        TopRow --> Card6[Unlearned Packets<br/>0]
        TopRow --> Card7[Notes<br/>]
        
        TopRow --> TrafficChart[Traffic Chart<br/>Time Series Visualization]
        TrafficChart --> ChartLib[Recharts Library]
        
        TrafficChart --> DeviceTable[Network Traffic Table]
        DeviceTable --> Col1[IP Column]
        DeviceTable --> Col2[Devices Column]
        DeviceTable --> Col3[Status Column]
        DeviceTable --> Col4[In/Out Traffic]
        
        DeviceTable --> AlertFeed[Alert Feed<br/>Real-time Updates]
        AlertFeed --> EmptyState[No alerts yet. Monitoring...]
    end
    
    style Header fill:#000000
    style Card1 fill:#ffffff
    style Card2 fill:#ffffff
    style Card3 fill:#ffffff
    style TrafficChart fill:#f5f5f5
    style DeviceTable fill:#ffffff
    style AlertFeed fill:#ffffff
```

## 3. Frontend Component Hierarchy

```mermaid
graph TB
    subgraph "App Component Tree"
        App[App.tsx<br/>Root Component]
        
        App --> AuthProvider[Auth Context Provider]
        AuthProvider --> ThemeProvider[MUI Theme Provider]
        ThemeProvider --> RouterProvider[Router Provider]
        
        RouterProvider --> Layout[Layout Component]
        
        Layout --> Navbar[Navigation Bar]
        Navbar --> Logo[QFLARE Logo]
        Navbar --> NavLinks[Navigation Links]
        Navbar --> UserMenu[User Menu Dropdown]
        Navbar --> NotifBell[Notification Bell]
        
        Layout --> Sidebar[Sidebar Menu]
        Sidebar --> DashLink[Dashboard]
        Sidebar --> PrivLink[Privacy]
        Sidebar --> SecLink[Security]
        Sidebar --> TrainLink[Training]
        Sidebar --> AdminLink[Admin]
        
        Layout --> MainContent[Main Content Area]
        
        MainContent --> DashPage[Dashboard Page]
        MainContent --> PrivPage[Privacy Page]
        MainContent --> SecPage[Security Page]
        MainContent --> TrainPage[Training Page]
        MainContent --> AdminPage[Admin Page]
        
        DashPage --> SummaryCards[Summary Cards Grid]
        SummaryCards --> MetricCard1[Packet Loss Card]
        SummaryCards --> MetricCard2[Traffic Card]
        SummaryCards --> MetricCard3[Bandwidth Card]
        
        DashPage --> TrafficViz[Traffic Visualization]
        TrafficViz --> LineChart[Line Chart Component]
        
        DashPage --> DeviceList[Device List Table]
        DeviceList --> TableHead[Table Header]
        DeviceList --> TableBody[Table Body]
        DeviceList --> TableRow[Table Rows]
        
        DashPage --> AlertSection[Alert Feed Section]
        AlertSection --> AlertList[Alert List]
        AlertSection --> AlertItem[Individual Alerts]
    end
    
    style App fill:#e3f2fd
    style Layout fill:#fff3e0
    style Navbar fill:#000000
    style Sidebar fill:#f5f5f5
    style MainContent fill:#ffffff
    style DashPage fill:#e8f5e9
    style PrivPage fill:#f3e5f5
    style SecPage fill:#ffebee
    style TrainPage fill:#fff9c4
    style AdminPage fill:#fce4ec
```

## 4. Data Flow - Real-time Dashboard Updates

```mermaid
sequenceDiagram
    participant User as User Browser
    participant React as React App
    participant Redux as Redux Store
    participant API as API Service
    participant Backend as FastAPI Backend
    participant Redis as Redis Cache
    participant DB as PostgreSQL
    
    Note over User,DB: Initial Page Load
    
    User->>React: Navigate to Dashboard
    React->>Redux: Dispatch fetchDashboardData()
    Redux->>API: GET /api/dashboard/stats
    API->>Backend: HTTP Request + JWT Token
    Backend->>Redis: Check cached data
    
    alt Cache Hit
        Redis-->>Backend: Return cached data
    else Cache Miss
        Backend->>DB: Query database
        DB-->>Backend: Return fresh data
        Backend->>Redis: Update cache (TTL: 30s)
    end
    
    Backend-->>API: JSON Response
    API-->>Redux: Update store
    Redux-->>React: Re-render components
    React-->>User: Display dashboard
    
    Note over User,DB: Real-time Updates
    
    loop Every 30 seconds
        React->>API: Auto-refresh data
        API->>Backend: GET /api/dashboard/stats
        Backend->>Redis: Get latest data
        Redis-->>Backend: Return data
        Backend-->>API: JSON Response
        API->>Redux: Update store
        Redux->>React: Re-render
        React->>User: Update UI
    end
    
    Note over User,DB: User Action
    
    User->>React: Click "Refresh" button
    React->>Redux: Dispatch refreshDashboard()
    Redux->>API: GET /api/dashboard/stats?force=true
    API->>Backend: Force refresh request
    Backend->>DB: Query latest data
    DB-->>Backend: Fresh data
    Backend->>Redis: Update cache
    Backend-->>API: JSON Response
    API->>Redux: Update store
    Redux->>React: Show success notification
    React->>User: Display updated data
```

## 5. Redux State Management Architecture

```mermaid
graph TB
    subgraph "Redux Store"
        Store[Redux Store<br/>Global State]
        
        Store --> Auth[Auth Slice]
        Store --> Dashboard[Dashboard Slice]
        Store --> Privacy[Privacy Slice]
        Store --> Security[Security Slice]
        Store --> Training[Training Slice]
        Store --> UI[UI Slice]
        Store --> Notifications[Notifications Slice]
    end
    
    subgraph "Auth Slice"
        Auth --> User[user: User | null]
        Auth --> Token[token: string | null]
        Auth --> IsAuth[isAuthenticated: boolean]
        Auth --> Loading[loading: boolean]
        Auth --> Error[error: string | null]
    end
    
    subgraph "Dashboard Slice"
        Dashboard --> Stats[stats: DashboardStats]
        Dashboard --> Traffic[trafficData: TimeSeriesData[]]
        Dashboard --> Devices[devices: Device[]]
        Dashboard --> Alerts[alerts: Alert[]]
        Dashboard --> RefreshTime[lastRefresh: timestamp]
    end
    
    subgraph "UI Slice"
        UI --> IsOnline[isOnline: boolean]
        UI --> Refreshing[refreshing: boolean]
        UI --> SidebarOpen[sidebarOpen: boolean]
        UI --> Theme[theme: 'light' | 'dark']
        UI --> Snackbar[snackbar: {open, message, severity}]
    end
    
    subgraph "Actions"
        Store --> Actions[Action Creators]
        Actions --> SyncActions[Sync Actions<br/>SET_USER, TOGGLE_SIDEBAR]
        Actions --> AsyncActions[Async Thunks<br/>fetchDashboard, login]
    end
    
    subgraph "Components"
        Store --> Selectors[Selectors]
        Selectors --> Comp1[useSelector Hook]
        Comp1 --> DashComp[Dashboard Component]
        Comp1 --> PrivComp[Privacy Component]
        Comp1 --> SecComp[Security Component]
    end
    
    style Store fill:#e3f2fd
    style Auth fill:#fff3e0
    style Dashboard fill:#e8f5e9
    style UI fill:#f3e5f5
    style Actions fill:#ffebee
    style Selectors fill:#fce4ec
```

## 6. API Request/Response Flow with Error Handling

```mermaid
graph TB
    subgraph "Component"
        Comp[React Component]
        Comp --> UseEffect[useEffect Hook]
        UseEffect --> FetchData[fetchData Function]
    end
    
    subgraph "API Utility Layer"
        FetchData --> IsAuth{isAuthenticated?}
        IsAuth -->|No| ShowLogin[Redirect to Login]
        IsAuth -->|Yes| AuthFetch[authenticatedFetch]
        
        AuthFetch --> AddToken[Add Bearer Token]
        AddToken --> SetTimeout[Set Timeout 15s]
        SetTimeout --> Controller[AbortController]
        Controller --> FetchAPI[Fetch API]
    end
    
    subgraph "Request Handling"
        FetchAPI --> Network{Network OK?}
        Network -->|Offline| OfflineError[Offline Error]
        Network -->|Online| SendRequest[Send HTTP Request]
        
        SendRequest --> Backend[Backend Server]
        Backend --> CheckAuth{Token Valid?}
        CheckAuth -->|No| Return401[401 Unauthorized]
        CheckAuth -->|Yes| ProcessReq[Process Request]
        
        ProcessReq --> Success{Success?}
        Success -->|Yes| Return200[200 OK + Data]
        Success -->|No| Return500[500 Error]
    end
    
    subgraph "Response Handling"
        Return200 --> ParseJSON[Parse JSON]
        ParseJSON --> UpdateState[Update Redux State]
        UpdateState --> ReRender[Re-render Component]
        
        Return401 --> ClearToken[Clear Token]
        ClearToken --> ShowLogin
        
        Return500 --> ShowError[Show Error Snackbar]
        OfflineError --> ShowOffline[Show Offline Badge]
    end
    
    subgraph "Timeout Handling"
        FetchAPI --> TimeoutCheck{Timeout?}
        TimeoutCheck -->|Yes| AbortReq[Abort Request]
        AbortReq --> TimeoutError[Timeout Error]
        TimeoutError --> Retry{Retry?}
        Retry -->|Yes| RetryLogic[Exponential Backoff]
        RetryLogic --> FetchAPI
        Retry -->|No| ShowTimeout[Show Timeout Error]
    end
    
    style Comp fill:#e3f2fd
    style AuthFetch fill:#fff3e0
    style Backend fill:#e8f5e9
    style UpdateState fill:#c8e6c9
    style ShowError fill:#ffcdd2
    style ShowOffline fill:#ffccbc
    style ShowTimeout fill:#fff9c4
```

## 7. Material-UI Component Structure

```mermaid
graph TB
    subgraph "MUI Theme Configuration"
        Theme[Custom Theme]
        Theme --> Palette[Palette]
        Palette --> Primary[Primary: #000000 Black]
        Palette --> Secondary[Secondary: #ffffff White]
        Palette --> Error[Error: #f44336 Red]
        Palette --> Success[Success: #4caf50 Green]
        
        Theme --> Typography[Typography]
        Typography --> Font[Font: Montserrat]
        Typography --> H4[h4: 600 weight]
        Typography --> Body[body1: 400 weight]
    end
    
    subgraph "Layout Components"
        Layout[Box Container]
        Layout --> Grid[Grid System]
        Grid --> Row1[Grid Row - 12 columns]
        Grid --> Row2[Grid Row - Responsive]
        
        Row1 --> Card1[Card 1<br/>md=3 columns]
        Row1 --> Card2[Card 2<br/>md=3 columns]
        Row1 --> Card3[Card 3<br/>md=3 columns]
        Row1 --> Card4[Card 4<br/>md=3 columns]
        
        Row2 --> Chart[Paper + Chart<br/>md=8 columns]
        Row2 --> Table[Paper + Table<br/>md=4 columns]
    end
    
    subgraph "Interactive Components"
        Card1 --> CardContent[CardContent]
        CardContent --> Icon[Icon Component]
        CardContent --> Text[Typography]
        CardContent --> Number[Chip/Badge]
        
        Chart --> LineChart[LineChart - Recharts]
        LineChart --> XAxis[XAxis - Time]
        LineChart --> YAxis[YAxis - Value]
        LineChart --> Line[Line - Data]
        LineChart --> Tooltip[Tooltip]
        LineChart --> Legend[Legend]
        
        Table --> TableContainer[TableContainer]
        TableContainer --> TableHead[TableHead]
        TableContainer --> TableBody[TableBody]
        TableBody --> TableRow[TableRow]
        TableRow --> TableCell[TableCell]
    end
    
    subgraph "Action Components"
        Action[User Actions]
        Action --> Button[Button - Refresh]
        Action --> IconButton[IconButton - Menu]
        Action --> Badge[Badge - Notifications]
        Action --> Menu[Menu - Dropdown]
        Action --> Dialog[Dialog - Modals]
    end
    
    style Theme fill:#e3f2fd
    style Layout fill:#fff3e0
    style Card1 fill:#ffffff
    style Chart fill:#f5f5f5
    style Table fill:#ffffff
    style Action fill:#e8f5e9
```

## 8. Responsive Design Breakpoints

```mermaid
graph LR
    subgraph "Breakpoint System"
        XS[Extra Small<br/>xs: 0-600px<br/>Mobile Portrait]
        SM[Small<br/>sm: 600-900px<br/>Mobile Landscape]
        MD[Medium<br/>md: 900-1200px<br/>Tablet]
        LG[Large<br/>lg: 1200-1536px<br/>Desktop]
        XL[Extra Large<br/>xl: 1536px+<br/>Large Desktop]
    end
    
    subgraph "Layout Adjustments"
        XS --> MobileLayout[Mobile Layout<br/>1 Column<br/>Stacked Cards]
        SM --> TabletLayout[Tablet Layout<br/>2 Columns<br/>Collapsed Sidebar]
        MD --> DesktopLayout[Desktop Layout<br/>3-4 Columns<br/>Full Sidebar]
        LG --> WideLayout[Wide Layout<br/>4+ Columns<br/>Expanded Charts]
        XL --> UltraWideLayout[Ultra Wide<br/>5+ Columns<br/>Side-by-side Views]
    end
    
    subgraph "Component Behavior"
        MobileLayout --> HideElements[Hide: Sidebar, Complex Charts]
        MobileLayout --> SimplifyUI[Simplify: Cards to List]
        
        TabletLayout --> CollapseNav[Collapse: Navigation Icons Only]
        TabletLayout --> AdaptCharts[Adapt: Smaller Charts]
        
        DesktopLayout --> FullFeatures[Show: All Features]
        FullFeatures --> ComplexViz[Complex: Multi-line Charts]
        
        WideLayout --> EnhancedViz[Enhanced: Side-by-side Dashboards]
        UltraWideLayout --> SplitScreen[Split: Multiple Views]
    end
    
    style XS fill:#ffccbc
    style SM fill:#fff9c4
    style MD fill:#c8e6c9
    style LG fill:#b3e5fc
    style XL fill:#e1bee7
```

## 9. WebSocket Real-time Communication (Future Enhancement)

```mermaid
sequenceDiagram
    participant Client as React Client
    participant WS as WebSocket Connection
    participant Server as FastAPI Server
    participant Redis as Redis Pub/Sub
    participant ML as ML Training Process
    
    Note over Client,ML: Connection Establishment
    
    Client->>WS: Connect to ws://localhost:8000/ws
    WS->>Server: WebSocket handshake
    Server->>WS: Connection accepted
    WS->>Client: Connection established
    
    Note over Client,ML: Subscribe to Channels
    
    Client->>WS: Subscribe: training_updates
    WS->>Server: Register subscription
    Server->>Redis: SUBSCRIBE training_updates
    
    Client->>WS: Subscribe: security_alerts
    WS->>Server: Register subscription
    Server->>Redis: SUBSCRIBE security_alerts
    
    Note over Client,ML: Real-time Updates
    
    ML->>Server: Training round completed
    Server->>Redis: PUBLISH training_updates
    Redis->>Server: Notify subscribers
    Server->>WS: Send update message
    WS->>Client: Receive training update
    Client->>Client: Update dashboard (no polling!)
    
    loop Every 3 seconds
        ML->>Server: Byzantine detection result
        Server->>Redis: PUBLISH security_alerts
        Redis->>Server: Alert notification
        Server->>WS: Send alert
        WS->>Client: Show alert badge
        Client->>Client: Update security dashboard
    end
    
    Note over Client,ML: Disconnection
    
    Client->>WS: Close connection
    WS->>Server: Disconnect event
    Server->>Redis: UNSUBSCRIBE all
    Server->>WS: Connection closed
```

## 10. Complete Page Flow - User Journey

```mermaid
graph TB
    Start[User opens browser] --> LoadApp[Load React App]
    
    LoadApp --> CheckAuth{Has Valid<br/>JWT Token?}
    CheckAuth -->|No| LoginPage[Show Login Page]
    CheckAuth -->|Yes| LoadDash[Load Dashboard]
    
    LoginPage --> EnterCreds[Enter Credentials]
    EnterCreds --> SubmitLogin[Submit Login Form]
    SubmitLogin --> Validate{Valid<br/>Credentials?}
    
    Validate -->|No| ShowError[Show Error Message]
    ShowError --> LoginPage
    
    Validate -->|Yes| GetToken[Receive JWT Token]
    GetToken --> StoreToken[Store in localStorage]
    StoreToken --> LoadDash
    
    LoadDash --> FetchData[Fetch Dashboard Data]
    FetchData --> RenderUI[Render UI Components]
    
    RenderUI --> ShowCards[Display Summary Cards]
    RenderUI --> ShowChart[Display Traffic Chart]
    RenderUI --> ShowTable[Display Device Table]
    RenderUI --> ShowAlerts[Display Alert Feed]
    
    ShowCards --> UserAction{User Action?}
    
    UserAction -->|Click Refresh| RefreshData[Refresh All Data]
    RefreshData --> FetchData
    
    UserAction -->|Navigate| NavChoice{Where?}
    NavChoice -->|Privacy| PrivPage[Privacy Dashboard]
    NavChoice -->|Security| SecPage[Security Dashboard]
    NavChoice -->|Training| TrainPage[Training Dashboard]
    NavChoice -->|Admin| AdminPage[Admin Panel]
    
    PrivPage --> FetchPrivacy[Fetch Privacy Data]
    SecPage --> FetchSecurity[Fetch Security Data]
    TrainPage --> FetchTraining[Fetch Training Data]
    AdminPage --> FetchAdmin[Fetch Admin Data]
    
    FetchPrivacy --> RenderPrivacy[Render Privacy UI]
    FetchSecurity --> RenderSecurity[Render Security UI]
    FetchTraining --> RenderTraining[Render Training UI]
    FetchAdmin --> RenderAdmin[Render Admin UI]
    
    RenderPrivacy --> UserAction
    RenderSecurity --> UserAction
    RenderTraining --> UserAction
    RenderAdmin --> UserAction
    
    UserAction -->|Logout| Logout[Clear Token]
    Logout --> LoginPage
    
    UserAction -->|Close Tab| End[End Session]
    
    style LoginPage fill:#fff3e0
    style LoadDash fill:#e8f5e9
    style PrivPage fill:#f3e5f5
    style SecPage fill:#ffebee
    style TrainPage fill:#fff9c4
    style AdminPage fill:#fce4ec
    style ShowError fill:#ffcdd2
```

---

## Usage Instructions

### Integration with your NIDS Control Center Design:

1. **Color Scheme**: Matches the black & white professional design
2. **Layout**: Mimics the packet loss, traffic, and device list structure
3. **Real-time Updates**: Shows how data flows from backend to frontend
4. **Component Structure**: Maps to actual React components in your codebase

### Rendering:

```bash
# Install VS Code extension
code --install-extension bierner.markdown-mermaid

# Or use online editor
https://mermaid.live

# Export as PNG/SVG for documentation
```

### Customization:

```javascript
// Adjust diagram theme
%%{init: {'theme':'dark'}}%%

// Or use custom theme
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#000'}}}%%
```

---

*Generated for QFLARE Dashboard - November 1, 2025*
