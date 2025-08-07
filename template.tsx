"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { cva, type VariantProps } from "class-variance-authority";
import { 
  Search, 
  Filter, 
  Download, 
  ExternalLink, 
  Users, 
  Calendar, 
  BookOpen, 
  Network, 
  ArrowUpRight,
  FileText,
  Star,
  Eye,
  Share2,
  Bookmark,
  TrendingUp,
  Globe,
  Database,
  Brain,
  Zap
} from "lucide-react";

function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}

const tabsVariants = cva(
  "relative inline-flex items-center justify-center rounded-lg transition-all duration-300 w-full",
  {
    variants: {
      variant: {
        default: "bg-background border border-border",
        ghost: "bg-transparent",
        underline: "bg-transparent border-b border-border rounded-none",
      },
      size: {
        sm: "h-9 p-1",
        default: "h-10 p-1.5",
        lg: "h-12 p-2",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

const tabTriggerVariants = cva(
  "relative inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all duration-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 flex-1",
  {
    variants: {
      variant: {
        default: "text-muted-foreground hover:text-foreground data-[state=active]:text-primary-foreground",
        ghost: "text-muted-foreground hover:text-foreground hover:bg-accent data-[state=active]:text-primary-foreground data-[state=active]:bg-transparent",
        underline: "text-muted-foreground hover:text-foreground data-[state=active]:text-accent-foreground rounded-none",
      },
      size: {
        sm: "px-2.5 py-1 text-xs",
        default: "px-3 py-1.5 text-sm",
        lg: "px-4 py-2 text-base",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface TabItem {
  id: string;
  label?: string;
  icon?: React.ReactNode;
}

export interface TabsProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof tabsVariants> {
  items: TabItem[];
  defaultValue?: string;
  value?: string;
  onValueChange?: (value: string) => void;
  indicatorColor?: string;
}

const CustomTabs = React.forwardRef<HTMLDivElement, TabsProps>(
  (
    {
      className,
      variant,
      size,
      items,
      defaultValue,
      value,
      onValueChange,
      indicatorColor = "rgb(59, 130, 246)",
      ...props
    },
    ref
  ) => {
    const [activeValue, setActiveValue] = React.useState(
      value || defaultValue || items[0]?.id
    );
    const [activeTabBounds, setActiveTabBounds] = React.useState({
      left: 0,
      width: 0,
    });

    const tabRefs = React.useRef<(HTMLButtonElement | null)[]>([]);

    React.useEffect(() => {
      if (value !== undefined) {
        setActiveValue(value);
      }
    }, [value]);

    React.useEffect(() => {
      const activeIndex = items.findIndex(
        (item: TabItem) => item.id === activeValue
      );
      const activeTab = tabRefs.current[activeIndex];

      if (activeTab) {
        const tabRect = activeTab.getBoundingClientRect();
        const containerRect = activeTab.parentElement?.getBoundingClientRect();

        if (containerRect) {
          setActiveTabBounds({
            left: tabRect.left - containerRect.left,
            width: tabRect.width,
          });
        }
      }
    }, [activeValue, items]);

    const handleTabClick = (tabId: string) => {
      setActiveValue(tabId);
      onValueChange?.(tabId);
    };

    return (
      <div
        ref={ref}
        className={cn(tabsVariants({ variant, size }), className)}
        {...props}
      >
        <motion.div
          className={cn(
            "absolute z-10",
            variant === "underline"
              ? "bottom-0 h-0.5 rounded-none"
              : "top-1 bottom-1 rounded-md"
          )}
          style={{
            backgroundColor: variant === "underline" ? "rgb(15, 23, 42)" : indicatorColor,
          }}
          initial={false}
          animate={{
            left: activeTabBounds.left,
            width: activeTabBounds.width,
          }}
          transition={{
            type: "spring",
            stiffness: 400,
            damping: 30,
          }}
        />
        {items.map((item: TabItem, index: number) => {
          const isActive = activeValue === item.id;

          return (
            <button
              key={item.id}
              ref={(el) => {
                tabRefs.current[index] = el;
              }}
              className={cn(
                tabTriggerVariants({ variant, size }),
                "relative z-20 text-slate-600 data-[state=active]:text-white gap-2"
              )}
              data-state={isActive ? "active" : "inactive"}
              onClick={() => handleTabClick(item.id)}
              type="button"
            >
              {item.icon && <span className="[&_svg]:size-4">{item.icon}</span>}
              {item.label}
            </button>
          );
        })}
      </div>
    );
  }
);

CustomTabs.displayName = "CustomTabs";

export interface TabsContentProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
  activeValue?: string;
}

const TabsContent = React.forwardRef<HTMLDivElement, TabsContentProps>(
  ({ className, value, activeValue, children, ...props }, ref) => {
    const isActive = value === activeValue;

    if (!isActive) return null;

    return (
      <motion.div
        ref={ref}
        className={cn(
          "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          className
        )}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 4 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

TabsContent.displayName = "TabsContent";

interface PaperCardProps {
  title?: string;
  authors?: string[];
  abstract?: string;
  year?: number;
  citations?: number;
  venue?: string;
  tags?: string[];
  url?: string;
}

const PaperCard: React.FC<PaperCardProps> = ({
  title = "Attention Is All You Need",
  authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  abstract = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
  year = 2017,
  citations = 85432,
  venue = "NeurIPS",
  tags = ["Transformers", "Attention", "NLP"],
  url = "#"
}) => {
  return (
    <div className="group relative overflow-hidden rounded-xl bg-white border border-slate-200 hover:border-blue-300 transition-all duration-300 hover:shadow-lg hover:shadow-blue-100/50">
      <div className="p-6">
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-lg font-semibold text-slate-900 group-hover:text-blue-700 transition-colors line-clamp-2">
            {title}
          </h3>
          <button className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-slate-100">
            <ExternalLink className="w-4 h-4 text-slate-600" />
          </button>
        </div>
        
        <div className="flex items-center gap-4 text-sm text-slate-600 mb-3">
          <span className="flex items-center gap-1">
            <Users className="w-3.5 h-3.5" />
            {authors.slice(0, 2).join(", ")}
            {authors.length > 2 && ` +${authors.length - 2}`}
          </span>
          <span className="flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5" />
            {year}
          </span>
          <span className="flex items-center gap-1">
            <Star className="w-3.5 h-3.5" />
            {citations.toLocaleString()}
          </span>
        </div>

        <p className="text-sm text-slate-700 mb-4 line-clamp-3">{abstract}</p>

        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-1.5">
            {tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 text-xs font-medium bg-blue-50 text-blue-700 rounded-md"
              >
                {tag}
              </span>
            ))}
          </div>
          <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded">
            {venue}
          </span>
        </div>
      </div>
    </div>
  );
};

interface NetworkNodeProps {
  id: string;
  label: string;
  x: number;
  y: number;
  size: number;
  color: string;
}

const NetworkVisualization: React.FC = () => {
  const nodes: NetworkNodeProps[] = [
    { id: "1", label: "Transformers", x: 200, y: 150, size: 40, color: "#3b82f6" },
    { id: "2", label: "BERT", x: 100, y: 100, size: 30, color: "#10b981" },
    { id: "3", label: "GPT", x: 300, y: 100, size: 35, color: "#f59e0b" },
    { id: "4", label: "Attention", x: 200, y: 50, size: 25, color: "#ef4444" },
    { id: "5", label: "RNN", x: 50, y: 200, size: 20, color: "#8b5cf6" },
    { id: "6", label: "CNN", x: 350, y: 200, size: 20, color: "#ec4899" },
  ];

  const connections = [
    { from: "1", to: "2" },
    { from: "1", to: "3" },
    { from: "1", to: "4" },
    { from: "2", to: "4" },
    { from: "3", to: "4" },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900">Research Network</h3>
        <div className="flex gap-2">
          <button className="px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors">
            Expand
          </button>
          <button className="px-3 py-1.5 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors">
            Export
          </button>
        </div>
      </div>
      
      <div className="relative h-64 bg-slate-50 rounded-lg overflow-hidden">
        <svg className="w-full h-full">
          {connections.map((conn, index) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            if (!fromNode || !toNode) return null;
            
            return (
              <line
                key={index}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke="#cbd5e1"
                strokeWidth="2"
                className="opacity-60"
              />
            );
          })}
          
          {nodes.map((node) => (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={node.size / 2}
                fill={node.color}
                className="opacity-80 hover:opacity-100 cursor-pointer transition-opacity"
              />
              <text
                x={node.x}
                y={node.y + node.size / 2 + 15}
                textAnchor="middle"
                className="text-xs font-medium fill-slate-700"
              >
                {node.label}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
};

const MaterialResearchDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState("search");

  const tabItems: TabItem[] = [
    { id: "search", label: "Search", icon: <Search /> },
    { id: "papers", label: "Papers", icon: <FileText /> },
    { id: "network", label: "Network", icon: <Network /> },
    { id: "scholars", label: "Scholars", icon: <Users /> },
  ];

  const mockPapers = [
    {
      title: "Attention Is All You Need",
      authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
      abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
      year: 2017,
      citations: 85432,
      venue: "NeurIPS",
      tags: ["Transformers", "Attention", "NLP"],
    },
    {
      title: "BERT: Pre-training of Deep Bidirectional Transformers",
      authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
      abstract: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations...",
      year: 2018,
      citations: 67891,
      venue: "NAACL",
      tags: ["BERT", "Language Model", "Pre-training"],
    },
    {
      title: "Language Models are Few-Shot Learners",
      authors: ["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
      abstract: "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus...",
      year: 2020,
      citations: 45123,
      venue: "NeurIPS",
      tags: ["GPT-3", "Few-shot", "Language Models"],
    },
  ];

  const mockScholars = [
    {
      name: "Geoffrey Hinton",
      affiliation: "University of Toronto",
      hIndex: 168,
      citations: 456789,
      recentPapers: 23,
      expertise: ["Deep Learning", "Neural Networks", "AI"],
    },
    {
      name: "Yann LeCun",
      affiliation: "NYU & Meta",
      hIndex: 145,
      citations: 398765,
      recentPapers: 18,
      expertise: ["Computer Vision", "CNN", "Self-supervised Learning"],
    },
    {
      name: "Yoshua Bengio",
      affiliation: "University of Montreal",
      hIndex: 142,
      citations: 387654,
      recentPapers: 31,
      expertise: ["Deep Learning", "RNN", "Representation Learning"],
    },
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-900">Material Research RAG</h1>
              <p className="text-slate-600">Advanced research discovery and analysis platform</p>
            </div>
          </div>
        </div>

        {/* Tabs Navigation */}
        <div className="mb-6">
          <CustomTabs
            items={tabItems}
            value={activeTab}
            onValueChange={setActiveTab}
            className="bg-white border border-slate-200 shadow-sm"
          />
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          <TabsContent value="search" activeValue={activeTab}>
            <div className="space-y-6">
              {/* Search Interface */}
              <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h2 className="text-xl font-semibold text-slate-900 mb-4">Research Search</h2>
                <div className="space-y-4">
                  <div className="flex gap-3">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <input
                        type="text"
                        placeholder="Search papers, authors, or topics..."
                        className="w-full pl-10 pr-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                      />
                    </div>
                    <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium">
                      Search
                    </button>
                  </div>
                  
                  <div className="flex gap-3 flex-wrap">
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                      <Filter className="w-4 h-4" />
                      Filters
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                      <Calendar className="w-4 h-4" />
                      Date Range
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                      <TrendingUp className="w-4 h-4" />
                      Sort by Citations
                    </button>
                  </div>
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                  { label: "Total Papers", value: "2.4M", icon: FileText, color: "blue" },
                  { label: "Active Scholars", value: "156K", icon: Users, color: "green" },
                  { label: "Research Areas", value: "847", icon: Database, color: "purple" },
                  { label: "Citations", value: "45.2M", icon: Star, color: "orange" },
                ].map((stat, index) => (
                  <div key={index} className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-slate-600">{stat.label}</p>
                        <p className="text-2xl font-bold text-slate-900">{stat.value}</p>
                      </div>
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-${stat.color}-50`}>
                        <stat.icon className={`w-6 h-6 text-${stat.color}-600`} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="papers" activeValue={activeTab}>
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-slate-900">Research Papers</h2>
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                    <Download className="w-4 h-4" />
                    Export
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    <BookOpen className="w-4 h-4" />
                    Add Paper
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {mockPapers.map((paper, index) => (
                  <PaperCard key={index} {...paper} />
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="network" activeValue={activeTab}>
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-slate-900">Research Network</h2>
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                    <Zap className="w-4 h-4" />
                    Auto-layout
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                    <Share2 className="w-4 h-4" />
                    Share
                  </button>
                </div>
              </div>

              <NetworkVisualization />

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  { label: "Connected Papers", value: "1,247", change: "+12%" },
                  { label: "Research Clusters", value: "23", change: "+3%" },
                  { label: "Citation Links", value: "5,891", change: "+18%" },
                ].map((metric, index) => (
                  <div key={index} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-slate-600">{metric.label}</p>
                        <p className="text-xl font-bold text-slate-900">{metric.value}</p>
                      </div>
                      <span className="text-sm font-medium text-green-600">{metric.change}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="scholars" activeValue={activeTab}>
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-slate-900">Scholar Profiles</h2>
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                    <Globe className="w-4 h-4" />
                    Browse All
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    <Users className="w-4 h-4" />
                    Add Scholar
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                {mockScholars.map((scholar, index) => (
                  <div key={index} className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-lg transition-shadow">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-slate-200 rounded-full flex items-center justify-center">
                          <span className="text-slate-700 font-semibold">
                            {scholar.name.split(' ').map(n => n[0]).join('')}
                          </span>
                        </div>
                        <div>
                          <h3 className="font-semibold text-slate-900">{scholar.name}</h3>
                          <p className="text-sm text-slate-600">{scholar.affiliation}</p>
                        </div>
                      </div>
                      <button className="p-1.5 rounded-lg hover:bg-slate-100 transition-colors">
                        <Bookmark className="w-4 h-4 text-slate-600" />
                      </button>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="text-center">
                        <p className="text-lg font-bold text-slate-900">{scholar.hIndex}</p>
                        <p className="text-xs text-slate-600">h-index</p>
                      </div>
                      <div className="text-center">
                        <p className="text-lg font-bold text-slate-900">{(scholar.citations / 1000).toFixed(0)}K</p>
                        <p className="text-xs text-slate-600">Citations</p>
                      </div>
                      <div className="text-center">
                        <p className="text-lg font-bold text-slate-900">{scholar.recentPapers}</p>
                        <p className="text-xs text-slate-600">Papers</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-medium text-slate-700">Expertise</p>
                      <div className="flex flex-wrap gap-1.5">
                        {scholar.expertise.map((area, areaIndex) => (
                          <span
                            key={areaIndex}
                            className="px-2 py-1 text-xs font-medium bg-slate-100 text-slate-700 rounded"
                          >
                            {area}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
        </div>
      </div>
    </div>
  );
};

export default MaterialResearchDashboard;
