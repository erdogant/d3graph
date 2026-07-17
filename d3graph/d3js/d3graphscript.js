function d3graphscript(config = {
    // Default values
    width: 800,
    height: 600,
    charge: -250,
    distance: 0,
    directed: false,
    collision: 0.5,
    link_tension: 1,
    sticky: false,
    background_color: '#FFFFFF',
    node_text_inside: false,
    max_ticks: 300,
    label_zoom_threshold: 0.6,
    canvas_edge_threshold: 2000,
    density_grid_size: 40,
    density_blur: 8,
    density_opacity: 0.6,
    show_density: false,
    dark_mode: false,
    }) {
    
    //Constants for the SVG
    //var width = config.width;
    //var height = config.height;

    // Constants for the SVG
    var width = config.width;
    var height = config.height;
    // When height is null, fill the remaining viewport below the header.
    if (height === null) {
        var header = document.querySelector(".header-row");
        var headerHeight = header ? header.offsetHeight : 0;
        height = window.innerHeight - headerHeight;
    }

    var background_color = config.background_color || '#FFFFFF';
    var sticky = config.sticky || false;
    // Cap how many simulation ticks run before auto-stopping, instead of
    // letting a large graph's force layout cool down naturally over
    // thousands of ticks (each one re-running collision detection over
    // every node). Restarting the simulation (drag, slider changes) resets
    // this counter so it still settles again after each change.
    var maxTicks = (config.max_ticks !== undefined && config.max_ticks !== null) ? config.max_ticks : 300;
    var tickCount = 0;
    // Below this zoom scale, labels are unreadable anyway — hide them (via a
    // single CSS class, not per-element work) instead of rendering thousands
    // of illegible <text> nodes. They reappear once zoomed back in past it.
    var labelZoomThreshold = (config.label_zoom_threshold !== undefined && config.label_zoom_threshold !== null) ? config.label_zoom_threshold : 0.6;
    // Above this many visible edges, draw links on a <canvas> instead of as
    // SVG <line> elements. SVG per-element overhead (DOM node creation,
    // layout, GC) is what makes tens of thousands of edges freeze the page;
    // canvas just issues draw calls into a single bitmap each frame, so cost
    // stays flat regardless of edge count. Nodes stay SVG either way (far
    // fewer of them, and it keeps drag/click/tooltip interactivity simple).
    var canvasEdgeThreshold = (config.canvas_edge_threshold !== undefined && config.canvas_edge_threshold !== null) ? config.canvas_edge_threshold : 2000;
    var useCanvasEdges = false;
    var canvasEl, ctx;
    var currentTransform = { scale: 1, translate: [0, 0] };
    // Master on/off switch for edges (independent of the weight/component
    // sliders) — lets the user clear visual clutter on large graphs to see
    // node structure/clustering without redrawing or refiltering anything.
    var edgesVisible = true;

    // ---- STATS PANEL (recolor nodes by a network statistic) ----
    // node_pagerank / node_hits_hub / node_hits_authority are precomputed
    // server-side (networkx) and normalized to [0, 1]; the panel just picks
    // which one (if any) drives node fill color. null = original node colors.
    // 'network_clustering' is the one exception: instead of a precomputed
    // per-node number, it's derived client-side (connected components over
    // whatever edges are currently visible past the weight-threshold slider)
    // so it always reflects the network's live, filtered state rather than
    // a static server-side clustering. networkClusters maps node index ->
    // small integer cluster id, recomputed on every applyStatStyling() call
    // while that mode is active.
    var currentStatKey = null;
    var networkClusters = {};
    
    // ---- LAYOUT MENU ----
    // 'force' (default) is the existing physics simulation, unchanged.
    // 'circular' / 'grid' are static, one-shot arrangements: force is
    // stopped, positions are computed directly, and updatePositions() draws
    // them once (no tick loop is running to do it automatically).
    var currentLayout = 'force';

    // COLOR SCHEME
    var statColorScale = d3.scale.linear()
    .domain([0, 0.5, 1])
    .range(["#2166ac", "#f7f7f7", "#b2182b"]);

    // Categorical palette for network-clustering mode — one color per
    // connected-component id, cycling every 20 clusters.
    var clusterColorScale = d3.scale.category20();
    
    // While a stat is active, nodes/edges get a visibility boost (bigger nodes,
    // thicker/more opaque edges) so the highlighted structure reads clearly.
    var statHighlightActive = false;
    var EDGE_HIGHLIGHT_WIDTH_MULT = 1.8;
    var EDGE_HIGHLIGHT_OPACITY_MULT = 1.6;

    // ---- DENSITY (clustering heatmap) LAYER ----
    // Grid-binned node density, drawn on its own canvas beneath the edges and
    // nodes. Recomputed from live node positions every tick it's visible, so
    // it tracks the force layout as nodes settle/move — cheap since it's a
    // single O(nodes) pass, unlike the edge count this was never the
    // bottleneck. Color scheme adapts to dark mode (updated live via
    // window.d3graphSetDarkMode, called from the dark-mode toggle).
    var densityGridSize = (config.density_grid_size !== undefined && config.density_grid_size !== null) ? config.density_grid_size : 40;
    var densityBlur = (config.density_blur !== undefined && config.density_blur !== null) ? config.density_blur : 8;
    var densityOpacity = (config.density_opacity !== undefined && config.density_opacity !== null) ? config.density_opacity : 0.6;
    var densityVisible = config.show_density || false;
    var darkMode = config.dark_mode || false;
    var densityCanvasEl, densityCtx, densityOffscreen;
    
    // Set the body background color
    document.body.style.backgroundColor = background_color;
    
    //Set up the colour scale
    var color = d3.scale.category20();
    
    var force = d3.layout.force()
      .charge(config.charge)
      .linkDistance((d) => d.edge_distance || config.distance)
      //.linkDistance((d) => config.distance > 0 ? config.distance : d.edge_weight)
      .linkStrength(config.link_tension !== undefined ? config.link_tension : 1)
      .size([width, height]);
    
    // ---- DRAGGING ----
    // Sticky mode: dragstart fixes the node so the simulation stops pulling it.
    //   dragend keeps it pinned (dashed stroke indicator).
    //   Right-click a pinned node to release it back into the simulation.
    // Normal mode: standard free-drag behaviour is preserved.
    
    function dragstarted(d) {
      d3.event.sourceEvent.stopPropagation();
      d3.select(this).classed("dragging", true);
      if (sticky && currentLayout === 'force') {
        d.fixed = true;
        tickCount = 0;
        force.start();
      }
    }
    
    function dragged(d) {
      if (sticky) {
        d.x = d.px = d3.event.x;
        d.y = d.py = d3.event.y;
      } else {
        d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
      }
      // With the force simulation stopped (static layouts), there's no tick
      // to refresh connected edges/labels — update them directly here.
      if (currentLayout !== 'force') updatePositions();
      if (densityVisible) drawDensityLayer();
    }
    
    function dragended(d) {
      d3.select(this).classed("dragging", false);
      if (sticky) {
        // Keep the node fixed and apply a visual "pinned" cue (dashed border)
        d.fixed = true;
        d3.select(this).select(".node-shape")
          .style("stroke-dasharray", "4,2")
          .style("stroke-width", function(d) { return Math.max(parseFloat(d.node_size_edge) || 1, 2); });
      }
    }
    
    var drag = force.drag()
      .origin(function(d) { return d; })
      .on("dragstart", dragstarted)
      .on("drag", dragged)
      .on("dragend", dragended);
    
    // ---- END DRAGGING ----

    // =====================================================================
    // ---- SHAPE RENDERING ------------------------------------------------
    // Supported node_marker values:
    //   'circle'                        – SVG <circle>  (default)
    //   'ellipse'                       – SVG <ellipse> (wider than tall)
    //   'square'                        – square via <path>
    //   'rectangle'                     – wide rectangle via <path>
    //   'triangle' / 'triangle-up'      – upward equilateral triangle
    //   'triangle-down'                 – downward triangle
    //   'diamond'                       – rotated square
    //   'star'                          – 5-pointed star
    //   'hexagon'                       – regular hexagon
    //   'pentagon'                      – regular pentagon
    // =====================================================================

    /**
     * Returns an SVG path `d` string for polygon/star shapes, centred at (0,0).
     * Returns null for 'circle' and 'ellipse' (handled as native SVG elements).
     */
    function shapePathD(marker, r) {
      r = parseFloat(r) || 8;
      var m = (marker || 'circle').toLowerCase();
      switch (m) {

        case 'triangle':
        case 'triangle-up': {
          var h = r * Math.sqrt(3);
          return 'M 0,' + (-r) +
                 ' L ' + (h / 2) + ',' + (r / 2) +
                 ' L ' + (-h / 2) + ',' + (r / 2) + ' Z';
        }

        case 'triangle-down': {
          var h = r * Math.sqrt(3);
          return 'M 0,' + r +
                 ' L ' + (h / 2) + ',' + (-r / 2) +
                 ' L ' + (-h / 2) + ',' + (-r / 2) + ' Z';
        }

        case 'square': {
          var s = r * 1.4;
          return 'M ' + (-s) + ',' + (-s) +
                 ' L ' +   s + ',' + (-s) +
                 ' L ' +   s + ',' +   s  +
                 ' L ' + (-s) + ',' +   s  + ' Z';
        }

        case 'rect':
        case 'rectangle': {
          var w = r * 2.0, h = r * 1.0;
          return 'M ' + (-w) + ',' + (-h) +
                 ' L ' +   w + ',' + (-h) +
                 ' L ' +   w + ',' +   h  +
                 ' L ' + (-w) + ',' +   h  + ' Z';
        }

        case 'diamond': {
          var rx = r * 1.0, ry = r * 1.4;
          return 'M 0,' + (-ry) +
                 ' L '  + rx + ',0' +
                 ' L 0,' + ry +
                 ' L ' + (-rx) + ',0 Z';
        }

        case 'star': {
          var outerR = r * 1.3, innerR = r * 0.5, pts = [];
          for (var i = 0; i < 10; i++) {
            var angle = (Math.PI / 5) * i - Math.PI / 2;
            var rad   = (i % 2 === 0) ? outerR : innerR;
            pts.push(+(rad * Math.cos(angle)).toFixed(3) + ',' + +(rad * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        case 'hexagon': {
          var pts = [];
          for (var i = 0; i < 6; i++) {
            var angle = (Math.PI / 3) * i - Math.PI / 6;
            pts.push(+(r * 1.1 * Math.cos(angle)).toFixed(3) + ',' + +(r * 1.1 * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        case 'pentagon': {
          var pts = [];
          for (var i = 0; i < 5; i++) {
            var angle = (2 * Math.PI / 5) * i - Math.PI / 2;
            pts.push(+(r * 1.15 * Math.cos(angle)).toFixed(3) + ',' + +(r * 1.15 * Math.sin(angle)).toFixed(3));
          }
          return 'M ' + pts.join(' L ') + ' Z';
        }

        default:
          return null; // circle or ellipse — caller uses native SVG element
      }
    }

    /**
     * Appends the correct SVG shape child to every node <g>.
     * All shapes receive the CSS class "node-shape" for uniform selection.
     *   circle / ellipse  →  native SVG elements positioned via cx/cy in tick
     *   everything else   →  <path> positioned via transform="translate()" in tick
     */
    function appendShape(nodeSelection) {
      nodeSelection.each(function(d) {
        var g      = d3.select(this);
        var marker = (d.node_marker || 'circle').toLowerCase();
        var r      = parseFloat(d.node_size) || 8;
        var shape;

        if (marker === 'ellipse') {
          shape = g.append('ellipse')
            .attr('class', 'node-shape')
            .attr('rx', r * 1.6)
            .attr('ry', r);
        } else if (marker === 'circle' || shapePathD(marker, r) === null) {
          shape = g.append('circle')
            .attr('class', 'node-shape')
            .attr('r', r);
        } else {
          shape = g.append('path')
            .attr('class', 'node-shape')
            .attr('d', shapePathD(marker, r));
        }

        shape
          .style('fill',         d.node_color)
          .style('opacity',      d.node_opacity)
          .style('stroke-width', d.node_size_edge)
          .style('stroke',       d.node_color_edge);
      });
    }

    // ---- END SHAPE RENDERING ----

    // Layered container: canvas (background + edges, used only above
    // canvasEdgeThreshold) sits under a transparent SVG (nodes always render
    // here, for drag/click/tooltip interactivity). The background color lives
    // on the container so it shows through the transparent SVG either way.
    var container = d3.select("body").append("div")
      .attr("id", "graphContainer")
      .style("position", "relative")
      .style("width", width + "px")
      .style("height", height + "px")
      .style("background-color", background_color);

    // Density (clustering heatmap) layer — created first so it stacks
    // beneath both the edge canvas and the SVG node layer.
    densityCanvasEl = container.append("canvas")
      .attr("width", width)
      .attr("height", height)
      .style("position", "absolute")
      .style("top", 0)
      .style("left", 0)
      .style("pointer-events", "none")
      .style("display", densityVisible ? null : "none")
      .node();
    densityCtx = densityCanvasEl.getContext("2d");

    canvasEl = container.append("canvas")
      .attr("width", width)
      .attr("height", height)
      .style("position", "absolute")
      .style("top", 0)
      .style("left", 0)
      .style("pointer-events", "none")
      .node();
    ctx = canvasEl.getContext("2d");

    //Append a SVG to the container. Assign this SVG as an object to svg
    var svg = container.append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("position", "absolute")
      .style("top", 0)
      .style("left", 0)
      .style("background-color", "transparent")
      .call(d3.behavior.zoom().on("zoom", function () {
        svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
        // Single class toggle (cheap) rather than iterating every label on
        // every zoom/pan event — CSS handles hiding all descendant <text>.
        svg.classed("labels-hidden", d3.event.scale < labelZoomThreshold);
        currentTransform.scale = d3.event.scale;
        currentTransform.translate = d3.event.translate;
        drawCanvasEdges();
        drawDensityLayer();
      }))
      .on("dblclick.zoom", null)
      .append("g")

    // Draws all links onto the canvas layer, matching the SVG group's current
    // pan/zoom transform. No-ops when the graph is small enough to stay on SVG.
    function drawCanvasEdges() {
      if (!useCanvasEdges || !ctx || !edgesVisible) return;
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      ctx.setTransform(currentTransform.scale, 0, 0, currentTransform.scale, currentTransform.translate[0], currentTransform.translate[1]);
      for (var i = 0; i < graph.links.length; i++) {
        var d = graph.links[i];
        if (!d.source || !d.target || typeof d.source.x !== 'number') continue;
        ctx.beginPath();
        ctx.moveTo(d.source.x, d.source.y);
        ctx.lineTo(d.target.x, d.target.y);
        ctx.strokeStyle = d.edge_color || '#999';
        var baseOpacity = (d.edge_opacity !== undefined && d.edge_opacity !== null) ? d.edge_opacity : 0.6;
        var baseWidth = d.edge_width || 1;
        ctx.globalAlpha = statHighlightActive ? Math.min(1, baseOpacity * EDGE_HIGHLIGHT_OPACITY_MULT) : baseOpacity;
        ctx.lineWidth = statHighlightActive ? baseWidth * EDGE_HIGHLIGHT_WIDTH_MULT : baseWidth;
        ctx.setLineDash(d.edge_style === 'dashed' ? [6, 3] : d.edge_style === 'dotted' ? [1.5, 3] : []);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Bins the currently visible nodes' live (x, y) positions into a coarse
    // grid over their bounding box. O(nodes) — cheap enough to recompute
    // every tick, unlike the edge count this was never the bottleneck.
    function computeDensityGrid() {
      var nodesData = node.data();
      if (!nodesData.length) return null;

      var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (var i = 0; i < nodesData.length; i++) {
        var nx = nodesData[i].x, ny = nodesData[i].y;
        if (typeof nx !== 'number' || typeof ny !== 'number') continue;
        if (nx < minX) minX = nx;
        if (nx > maxX) maxX = nx;
        if (ny < minY) minY = ny;
        if (ny > maxY) maxY = ny;
      }
      if (minX === Infinity) return null;

      var padX = (maxX - minX) * 0.05 || 20;
      var padY = (maxY - minY) * 0.05 || 20;
      minX -= padX; maxX += padX; minY -= padY; maxY += padY;

      var w = (maxX - minX) || 1;
      var h = (maxY - minY) || 1;
      var cellsX, cellsY;
      if (w >= h) {
        cellsX = densityGridSize;
        cellsY = Math.max(1, Math.round(densityGridSize * h / w));
      } else {
        cellsY = densityGridSize;
        cellsX = Math.max(1, Math.round(densityGridSize * w / h));
      }
      var cellW = w / cellsX;
      var cellH = h / cellsY;

      var grid = new Float32Array(cellsX * cellsY);
      var maxCount = 0;
      for (var j = 0; j < nodesData.length; j++) {
        var d = nodesData[j];
        if (typeof d.x !== 'number' || typeof d.y !== 'number') continue;
        var cx = Math.min(cellsX - 1, Math.max(0, Math.floor((d.x - minX) / cellW)));
        var cy = Math.min(cellsY - 1, Math.max(0, Math.floor((d.y - minY) / cellH)));
        var idx = cy * cellsX + cx;
        grid[idx] += 1;
        if (grid[idx] > maxCount) maxCount = grid[idx];
      }

      return { grid: grid, cellsX: cellsX, cellsY: cellsY, cellW: cellW, cellH: cellH, minX: minX, minY: minY, maxCount: maxCount };
    }

    // t in [0, 1] (relative density) -> fill color. Light mode: yellow -> red
    // heat gradient. Dark mode: single-hue blue, so it reads well against a
    // dark background instead of clashing with it.
    function densityColor(t) {
      if (darkMode) {
        var l = 25 + t * 45; // 25%..70% lightness
        return 'hsl(210, 90%, ' + l + '%)';
      }
      var hue = 60 - t * 60; // 60=yellow -> 0=red
      return 'hsl(' + hue + ', 100%, 50%)';
    }

    // Draws the heatmap: unblurred cells go to an offscreen buffer first,
    // then get composited onto the visible canvas with a single blurred
    // drawImage — much cheaper than blurring each cell individually.
    function drawDensityLayer() {
      if (!densityVisible || !densityCtx) return;
      densityCtx.save();
      densityCtx.setTransform(1, 0, 0, 1, 0, 0);
      densityCtx.clearRect(0, 0, densityCanvasEl.width, densityCanvasEl.height);

      var data = computeDensityGrid();
      if (!data || data.maxCount <= 0) { densityCtx.restore(); return; }

      if (!densityOffscreen) densityOffscreen = document.createElement('canvas');
      densityOffscreen.width = densityCanvasEl.width;
      densityOffscreen.height = densityCanvasEl.height;
      var offCtx = densityOffscreen.getContext('2d');
      offCtx.clearRect(0, 0, densityOffscreen.width, densityOffscreen.height);
      offCtx.setTransform(currentTransform.scale, 0, 0, currentTransform.scale, currentTransform.translate[0], currentTransform.translate[1]);

      for (var cy = 0; cy < data.cellsY; cy++) {
        for (var cx = 0; cx < data.cellsX; cx++) {
          var count = data.grid[cy * data.cellsX + cx];
          if (count <= 0) continue;
          var t = count / data.maxCount;
          offCtx.fillStyle = densityColor(t);
          offCtx.globalAlpha = densityOpacity * t;
          offCtx.fillRect(data.minX + cx * data.cellW, data.minY + cy * data.cellH, data.cellW, data.cellH);
        }
      }

      densityCtx.filter = 'blur(' + densityBlur + 'px)';
      densityCtx.drawImage(densityOffscreen, 0, 0);
      densityCtx.filter = 'none';
      densityCtx.restore();
    }


    function applyEdgeVisibility() {
      svg.classed("edges-hidden", !edgesVisible);
      if (useCanvasEdges) {
        if (edgesVisible) {
          d3.select(canvasEl).style("display", null);
          drawCanvasEdges();
        } else {
          if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
          d3.select(canvasEl).style("display", "none");
        }
      } else {
        d3.select(canvasEl).style("display", "none");
      }
    }

    // Builds (or tears down) the SVG <line>/<text> elements for links, or
    // switches to canvas rendering, based on the current edge count. Shared
    // by the initial render and by restart() (slider-driven updates), so the
    // threshold is re-evaluated every time the visible edge set changes.
    function renderLinks() {
      useCanvasEdges = graph.links.length > canvasEdgeThreshold;

      if (useCanvasEdges) {
        // Canvas takes over — drop any SVG link/link-text DOM entirely.
        link = link.data([]);
        link.exit().remove();
        linkText = linkText.data([]);
        linkText.exit().remove();
      } else {
        if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        link = link.data(graph.links);
        link.exit().remove();
        var linkEnter = link.enter().insert("line", ".node")
          .attr("class", "link")
          .attr('marker-start', function(d) { return 'url(#marker_' + d.marker_start + ')' });
        linkEnter.append("title").text(function(d) { return d.tooltip; });
        link.attr("marker-end", function(d) {
          if (config.directed) { return 'url(#marker_' + d.marker_end + ')' } });
        link.style("stroke-width", function(d) { return d.edge_width; });
        link.style("stroke", function(d) { return d.edge_color; });
        link.style("stroke-dasharray", function(d) { return d.edge_style; });
        link.style("opacity", function(d) { return d.edge_opacity; });

        linkText = linkText.data(graph.links);
        linkText.exit().remove();
        linkText.enter().append("text")
          .attr("class", "link-text")
          .attr("font-size", function(d) { return d.label_fontsize + "px"; })
          .style("fill", function(d) { return d.label_color; })
          .style("font-family", "Arial")
          .text(function(d) { return d.label; });
      }

      applyEdgeVisibility();
    }
    
    graphRec = JSON.parse(JSON.stringify(graph));   // Full, unfiltered copy — used by the slider to restore edges later

    // Apply the initial slider threshold BEFORE building any DOM elements.
    // Without this, the browser would create a <line>/<title>/<text> for every
    // single edge (e.g. 100,000+) and immediately delete most of them once the
    // slider filter ran — doubling the work and freezing the page in the meantime.
    (function applyInitialThreshold() {
      var initialThresh = {{ SET_SLIDER }};
      graph.links = graph.links.filter(function(d) { return d.edge_weight > initialThresh; });
    })();

    //Creates the graph data structure out of the json data
    force.nodes(graph.nodes)
      .links(graph.links)
      .start();
    
    // Create empty node/link/link-text selections. renderLinks(), called
    // below, decides whether links go to SVG or canvas and populates
    // link/linkText accordingly.
    var link = svg.selectAll(".link").data([]);
    var linkText = svg.selectAll(".link-text").data([]);
    renderLinks();

    //Do the same with the circles for the nodes
    var node = svg.selectAll(".node")
      .data(graph.nodes)
      .enter().append("g")
      .attr("class", "node")
      .call(drag)
      .on('dblclick', connectedNodes); // HIGHLIGHT ON/OFF
    
    // Right-click handler: release a pinned node back into the simulation (sticky mode only)
    if (sticky) {
      node.on('contextmenu', function(d) {
        d3.event.preventDefault();
        d.fixed = false;
        // Remove pinned visual cue
        d3.select(this).select(".node-shape")
          .style("stroke-dasharray", null)
          .style("stroke-width", function(d) { return d.node_size_edge; });
        tickCount = 0;
        force.resume();
      });
    }
    
    {{ CLICK_COMMENT }} node.on('click', color_on_click); // ON CLICK HANDLER

    // Append the correct shape per node (circle, ellipse, rect, triangle, etc.)
    appendShape(node);
    
    // Text in nodes
    node.append("text")
      .attr("dx",  function(d) { return config.node_text_inside ? 0 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return config.node_text_inside ? "middle" : "start"; })
      .text(function(d) { return d.node_name; }) // NODE-TEXT
      .style("font-size", function(d) {
        if (config.node_text_inside) {
          // Auto-shrink font so the label fits within the circle diameter
          var r = parseFloat(d.node_size) || 8;
          var label = d.node_name || "";
          // Approximate: 0.6 * fontsize ≈ char width; fit label in 2*r with some padding
          var maxFontSize = parseFloat(d.node_fontsize) || 12;
          var fitSize = Math.min(maxFontSize, (1.8 * r) / Math.max(label.length * 0.6, 1));
          return Math.max(fitSize, 6) + "px";
        }
        return d.node_fontsize + "px"; // NODE FONT SIZE
      })
      .style("fill", function(d) {return d.node_fontcolor;}) // NODE FONT COLOR
      .style("font-family", "monospace")
      .style("pointer-events", "none");
    
    let showInHover = ["node_tooltip"]; // Tooltip
    node.append("title")
        .text((d) => Object.keys(d)
            .filter((key) => showInHover.indexOf(key) !== -1)
            .map((key) => `${d[key]}`)
            .join('\n')
        )
    
    //Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
    // Draws nodes/links/labels at their current (d.x, d.y). Called every
    // force tick, and also directly by static layouts (circular/grid) that
    // set positions once instead of simulating — those need a render call
    // since there's no tick to trigger it.
    function updatePositions() {
      if (useCanvasEdges) {
        drawCanvasEdges();
      } else {
        link.attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });
      }

      // Position each shape according to its SVG element type:
      //   circle / ellipse  →  cx / cy attributes
      //   path              →  transform translate(x, y)
      // Scoped to the bound `node` selection instead of re-querying the
      // entire DOM every tick (was: d3.selectAll(".node-shape")).
      node.select(".node-shape").each(function(d) {
        var el  = d3.select(this);
        var tag = this.tagName.toLowerCase();
        if (tag === 'circle' || tag === 'ellipse') {
          el.attr("cx", d.x).attr("cy", d.y);
        } else {
          el.attr("transform", "translate(" + d.x + "," + d.y + ")");
        }
      });

      // Scoped to node labels only (was: d3.selectAll("text"), which also
      // re-matched every link-text element on every tick).
      node.select("text").attr("x", function(d) { return d.x; })
        .attr("y", function(d) { return d.y; })
      linkText.attr("x", function(d) { return (d.source.x + d.target.x) / 2; })  // ADD TEXT ON THE EDGES (PART 2/2)
        .attr("y", function(d) { return (d.source.y + d.target.y) / 2; })
        .attr("text-anchor", "middle");

      if (densityVisible) drawDensityLayer();
    }

    force.on("tick", function() {
      // Auto-stop after maxTicks so a large graph doesn't keep re-running
      // collision detection / layout math for thousands of frames while
      // settling. maxTicks <= 0 disables the cap (run to natural cooldown).
      if (maxTicks > 0 && ++tickCount > maxTicks) {
        force.stop();
        return;
      }

      updatePositions();
      node.each(collide(config.collision)); //COLLISION DETECTION. High means a big fight to get untouchable nodes (default=0.5)

  });

  // --------- MARKER FOR EDGE ENDINGS -----------

  var data_marker = [
    { id: 0, name: 'circle', path: 'M 0, 0  m -5, 0  a 5,5 0 1,0 10,0  a 5,5 0 1,0 -10,0', viewbox: '-6 -6 12 12' }
  , { id: 1, name: 'square', path: 'M 0,0 m -5,-5 L 5,-5 L 5,5 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 2, name: 'arrow', path: 'M 0,0 m -5,-5 L 5,0 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 3, name: 'stub', path: 'M 0,0 m -1,-5 L 1,-5 L 1,5 L -1,5 Z', viewbox: '-1 -5 2 10' }
  ]

  svg.append("defs").selectAll("marker")
    .data(data_marker)
    .enter()
    .append('svg:marker')
      .attr('id', function(d){ return 'marker_' + d.name})
      .attr('markerHeight', 10)
      .attr('markerWidth', 10)
      .attr("markerUnits", "userSpaceOnUse")                   // Fix marker width
      .attr('orient', 'auto')
      .attr('refX', 15)                                        // Offset marker-end
      .attr('refY', 0)
      .attr('viewBox', function(d){ return d.viewbox })
      .append('svg:path')
        .attr('d', function(d){ return d.path })               // Marker type
        .style("fill", '#808080')                              // Marker color
        .style("stroke", '#808080')                            // Marker edge-color
        .style("opacity", 0.95)                                // Marker opacity
        .style("stroke-width", 1);                             // Marker edge thickness

  // --------- END MARKER -----------


  // collision detection

  var padding = 1, // separation between circles
    radius = 8;

  function collide(alpha) {
    var quadtree = d3.geom.quadtree(graph.nodes);
    return function(d) {
      var rb = 2 * radius + padding,
        nx1 = d.x - rb,
        nx2 = d.x + rb,
        ny1 = d.y - rb,
        ny2 = d.y + rb;
      quadtree.visit(function(quad, x1, y1, x2, y2) {
        if (quad.point && (quad.point !== d)) {
          var x = d.x - quad.point.x,
            y = d.y - quad.point.y,
            l = Math.sqrt(x * x + y * y);
          if (l < rb) {
            l = (l - rb) / l * alpha;
            d.x -= x *= l;
            d.y -= y *= l;
            quad.point.x += x;
            quad.point.y += y;
          }
        }
        return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
      });
    };
  }
  // collision detection end


  //Toggle stores whether the highlighting is on
  var toggle = 0;
  //Create an array logging what is connected to what
  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function(d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });
  //This function looks up whether a pair are neighbours
  function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
  }


  // COLOR ON CLICK
  function color_on_click() {

  // First restore the currently active base styling:
  // - statistic colors/sizes when a radio metric is active
  // - original node styling when no statistic is active
  applyStatStyling();

  // Restore pinned cue after applyStatStyling()
  node.select(".node-shape")
    .style("stroke-dasharray", function(d) {
      return (sticky && d.fixed) ? "4,2" : null;
    });

  // Apply click styling only to the selected node
  var clickedShape = d3.select(this).select(".node-shape");

  clickedShape
    .style("fill", {{ CLICK_FILL }})
    .style("stroke", "{{ CLICK_STROKE }}")
    .style("stroke-width", {{ CLICK_STROKEW }});

  var shapeNode = clickedShape.node();
  if (!shapeNode) return;

  var tag = shapeNode.tagName.toLowerCase();

  clickedShape.each(function(d) {
    var el = d3.select(this);
    var baseR = parseFloat(d.node_size) || 8;

    // Preserve the statistic-driven size before applying click enlargement
    var statValue = currentStatKey ? d[currentStatKey] : null;
    var hasStatValue =
      currentStatKey &&
      typeof statValue === "number" &&
      !isNaN(statValue);

    var statScale = hasStatValue
      ? 1.2 + statValue * 1.3
      : 1;

    var clickScale = statScale * {{ CLICK_SIZE }};

    if (tag === "circle") {
      el.attr("r", baseR * clickScale);

    } else if (tag === "ellipse") {
      el
        .attr("rx", baseR * 1.6 * clickScale)
        .attr("ry", baseR * clickScale);

    } else if (tag === "path") {
      el.attr(
        "d",
        shapePathD(d.node_marker, baseR * clickScale)
      );
    }
  });
}


  function connectedNodes() {
    if (toggle == 0) {
      //Reduce the opacity of all but the neighbouring nodes
      d = d3.select(this).node().__data__;
      node.style("opacity", function(o) {
        return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
      });
      link.style("opacity", function(o) {
        return d.index == o.source.index | d.index == o.target.index ? 1 : 0.1;
      });
      toggle = 1;
    } else {
      //Put them back to opacity=1
      node.style("opacity", 0.95);
      link.style("opacity", 1);

      toggle = 0;
    }
  }


  //adjust threshold
  function threshold() {
    let thresh = this.value;

    // Reflect the current slider value in the header label. Kept inside
    // threshold() itself (not a separate handler) so it stays in sync with
    // every path that sets the network state: the initial load call below,
    // and the "change" listener — no risk of the two drifting apart.
    var thresholdValueEl = document.getElementById('thresholdValue');
    if (thresholdValueEl) thresholdValueEl.textContent = thresh;

    graph.links.splice(0, graph.links.length);
    linkText = linkText.data([]);   // CLEAR EDGE-LABELS: Clear the linkText selection
    linkText.exit().remove();       // CLEAR EDGE-LABELS: Clear the linkText elements from the DOM

    for (var i = 0; i < graphRec.links.length; i++) {
      if (graphRec.links[i].edge_weight > thresh) {
        graph.links.push(graphRec.links[i]);
      }
    }
    restart();
  }

  // Set the initial value of the slider to the user-defined threshold
  document.getElementById('thresholdSlider').value = {{ SET_SLIDER }};
  // Call the threshold function to set the network state
  threshold.call(document.getElementById('thresholdSlider'));

  d3.select("#thresholdSlider").on("change", threshold);

  // Live-update the label while dragging, without paying for a full
  // restart() (network refilter + recompute of whichever stat is active)
  // on every intermediate value — that still only happens on "change"
  // (drag release), same as before.
  var thresholdValueLiveEl = document.getElementById('thresholdValue');
  var thresholdSliderEl = document.getElementById('thresholdSlider');
  if (thresholdValueLiveEl && thresholdSliderEl) {
    thresholdSliderEl.addEventListener('input', function() {
      thresholdValueLiveEl.textContent = this.value;
    });
  }

  // Master edges on/off toggle — independent of the weight slider, purely
  // visual, doesn't touch the underlying filtered data.
  var edgeToggleBtn = document.getElementById('edgeToggleButton');
  if (edgeToggleBtn) {
    edgeToggleBtn.addEventListener('click', function() {
      edgesVisible = !edgesVisible;
      edgeToggleBtn.textContent = edgesVisible ? 'Hide Edges' : 'Show Edges';
      applyEdgeVisibility();
    });
  }

  // Density (clustering heatmap) layer toggle.
  var densityToggleBtn = document.getElementById('densityToggleButton');
  var densityPanelEl = document.getElementById('densityPanel');
  if (densityToggleBtn) {
    densityToggleBtn.addEventListener('click', function() {
      densityVisible = !densityVisible;
      densityToggleBtn.textContent = densityVisible ? 'Hide Density' : 'Show Density';
      d3.select(densityCanvasEl).style("display", densityVisible ? null : "none");
      if (densityVisible) {
        drawDensityLayer();
      } else {
        densityCtx.setTransform(1, 0, 0, 1, 0, 0);
        densityCtx.clearRect(0, 0, densityCanvasEl.width, densityCanvasEl.height);
      }
      // The density tuning panel (grid size / blur / opacity) is only
      // useful while the layer is actually visible.
      if (densityPanelEl) densityPanelEl.style.display = densityVisible ? '' : 'none';
      // Re-stack the remaining side panels now that one changed size/visibility.
      if (typeof positionSidePanels === 'function') positionSidePanels();
    });
    // Reflect the initial show_density state on load, in case it started true.
    if (densityPanelEl) densityPanelEl.style.display = densityVisible ? '' : 'none';
  }

  // Called from the dark-mode toggle (outside this function's scope, in the
  // page's own script) so the density color scheme switches live instead of
  // only reflecting whatever mode the page loaded in.
  window.d3graphSetDarkMode = function(isDark) {
    darkMode = !!isDark;
    if (densityVisible) drawDensityLayer();
  };

  // ---- DYNAMIC STAT RECOMPUTATION ----
  // PageRank / HITS / degree / closeness / betweenness centrality arrive
  // precomputed server-side (networkx), since that's the only graph that
  // exists at render time. But the weight-threshold slider changes which
  // edges are actually "in" the network, and a statistic computed on
  // since-removed edges is stale. So these are recomputed client-side from
  // graph.links (the currently filtered edge set) every time
  // applyStatStyling() runs — i.e. on stat selection AND on every slider
  // change (restart() -> applyStatStyling()) — overwriting d[<stat key>]
  // on each node so the rest of the styling pipeline needs no changes.
  var DYNAMIC_STAT_KEYS = ['node_pagerank', 'node_hits_hub', 'node_hits_authority',
    'node_degree_centrality', 'node_closeness_centrality', 'node_betweenness_centrality'];

  // Adjacency built fresh from the CURRENT graph.nodes/graph.links.
  //
  // IMPORTANT: this always follows edges source -> target, never
  // symmetrized — regardless of config.directed. config.directed only
  // controls whether arrowheads are drawn (it's a display setting); the
  // underlying graph object these stats need to match is the one built
  // server-side by make_graph(), which is unconditionally an nx.DiGraph
  // (see d3graph.py). Treating the network as undirected here (as an
  // earlier version of this code did whenever config.directed was false,
  // which is also the default) was the actual bug behind mismatched
  // HITS hub/authority and skewed degree/closeness/betweenness scores —
  // it silently added a reverse edge for every edge, which networkx's
  // computation never does.
  function buildAdjacency() {
    var indices = graph.nodes.map(function(d) { return d.index; });
    var out = {}, inn = {};
    indices.forEach(function(i) { out[i] = []; inn[i] = []; });

    graph.links.forEach(function(l) {
      var s = (l.source && l.source.index !== undefined) ? l.source.index : l.source;
      var t = (l.target && l.target.index !== undefined) ? l.target.index : l.target;
      if (out[s] === undefined || out[t] === undefined) return; // safety
      out[s].push(t);
      inn[t].push(s);
    });

    return { indices: indices, out: out, inn: inn, n: indices.length };
  }

  function computeDegreeCentrality(adj) {
    var result = {};
    var n = adj.n;
    adj.indices.forEach(function(i) {
      // Matches networkx's degree_centrality on a DiGraph: total (in + out)
      // degree / (n - 1).
      var deg = adj.out[i].length + adj.inn[i].length;
      result[i] = (n > 1) ? deg / (n - 1) : 0;
    });
    return result;
  }

  function bfsDistances(adj, sourceIdx, neighborsOf) {
    var dist = {};
    dist[sourceIdx] = 0;
    var queue = [sourceIdx];
    var head = 0;
    while (head < queue.length) {
      var u = queue[head++];
      var neighbors = neighborsOf[u];
      for (var k = 0; k < neighbors.length; k++) {
        var v = neighbors[k];
        if (dist[v] === undefined) {
          dist[v] = dist[u] + 1;
          queue.push(v);
        }
      }
    }
    return dist;
  }

  function computeClosenessCentrality(adj) {
    var result = {};
    var n = adj.n;
    adj.indices.forEach(function(i) {
      // networkx's closeness_centrality reverses directed graphs before
      // computing shortest paths, so a node's score reflects how reachable
      // it is FROM the rest of the network (incoming paths), not how far
      // it can reach outward. Traverse via `inn` (in-edges) to match —
      // using `out` here was inverting the metric for any graph with real
      // directionality.
      var dist = bfsDistances(adj, i, adj.inn);
      var reachable = Object.keys(dist).length; // includes i itself
      var totalDist = 0;
      for (var key in dist) totalDist += dist[key];
      if (reachable > 1 && totalDist > 0 && n > 1) {
        // Wasserman-Faust "improved" formula (networkx default): scales by
        // the fraction of the graph actually reached, so a node stuck in a
        // small component doesn't get an inflated score just for being
        // close to the few nodes it can see.
        result[i] = ((reachable - 1) / totalDist) * ((reachable - 1) / (n - 1));
      } else {
        result[i] = 0;
      }
    });
    return result;
  }

  // Brandes' algorithm, unweighted — O(V*E), single pass computes
  // betweenness for every node at once.
  function computeBetweennessCentrality(adj) {
    var n = adj.n;
    var betweenness = {};
    adj.indices.forEach(function(i) { betweenness[i] = 0; });

    adj.indices.forEach(function(s) {
      var S = [];
      var P = {}, sigma = {}, dist = {};
      adj.indices.forEach(function(i) { P[i] = []; sigma[i] = 0; dist[i] = -1; });
      sigma[s] = 1;
      dist[s] = 0;
      var queue = [s];
      var head = 0;
      while (head < queue.length) {
        var v = queue[head++];
        S.push(v);
        var neighbors = adj.out[v];
        for (var k = 0; k < neighbors.length; k++) {
          var w = neighbors[k];
          if (dist[w] < 0) {
            queue.push(w);
            dist[w] = dist[v] + 1;
          }
          if (dist[w] === dist[v] + 1) {
            sigma[w] += sigma[v];
            P[w].push(v);
          }
        }
      }
      var delta = {};
      adj.indices.forEach(function(i) { delta[i] = 0; });
      while (S.length) {
        var w = S.pop();
        for (var k = 0; k < P[w].length; k++) {
          var v = P[w][k];
          delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]);
        }
        if (w !== s) betweenness[w] += delta[w];
      }
    });

    // Normalize (networkx defaults, normalized=True, directed graph): scale
    // by 1/((n-1)(n-2)) — no halving, since directed pairs (s,t) and (t,s)
    // are distinct and each is only ever visited once as a source here.
    var scale = (n > 2) ? 1 / ((n - 1) * (n - 2)) : 0;
    adj.indices.forEach(function(i) { betweenness[i] = betweenness[i] * scale; });

    return betweenness;
  }

  // Power-iteration PageRank (damping 0.85), with dangling nodes (no
  // out-edges) leaking their rank uniformly to every node, same as
  // networkx's default handling.
  function computePageRank(adj) {
    var n = adj.n;
    if (n === 0) return {};
    var alpha = 0.85, maxIter = 100, tol = 1e-8;
    var rank = {};
    adj.indices.forEach(function(i) { rank[i] = 1 / n; });
    var outDegree = {};
    adj.indices.forEach(function(i) { outDegree[i] = adj.out[i].length; });

    for (var iter = 0; iter < maxIter; iter++) {
      var newRank = {};
      adj.indices.forEach(function(i) { newRank[i] = (1 - alpha) / n; });

      var danglingSum = 0;
      adj.indices.forEach(function(i) { if (outDegree[i] === 0) danglingSum += rank[i]; });
      adj.indices.forEach(function(i) { newRank[i] += alpha * danglingSum / n; });

      adj.indices.forEach(function(u) {
        if (outDegree[u] === 0) return;
        var share = alpha * rank[u] / outDegree[u];
        adj.out[u].forEach(function(v) { newRank[v] += share; });
      });

      var diff = 0;
      adj.indices.forEach(function(i) { diff += Math.abs(newRank[i] - rank[i]); });
      rank = newRank;
      if (diff < tol) break;
    }

    return rank;
  }

  // Power-iteration HITS. Hub and authority are computed together since
  // they're mutually dependent either way.
  function computeHITS(adj) {
    var hub = {}, auth = {};
    adj.indices.forEach(function(i) { hub[i] = 1; auth[i] = 1; });
    var maxIter = 100, tol = 1e-8;

    for (var iter = 0; iter < maxIter; iter++) {
      var newAuth = {};
      adj.indices.forEach(function(i) { newAuth[i] = 0; });
      adj.indices.forEach(function(u) {
        adj.out[u].forEach(function(v) { newAuth[v] += hub[u]; });
      });
      var authNorm = Math.sqrt(adj.indices.reduce(function(s, i) { return s + newAuth[i] * newAuth[i]; }, 0)) || 1;
      adj.indices.forEach(function(i) { newAuth[i] /= authNorm; });

      var newHub = {};
      adj.indices.forEach(function(i) { newHub[i] = 0; });
      adj.indices.forEach(function(u) {
        adj.out[u].forEach(function(v) { newHub[u] += newAuth[v]; });
      });
      var hubNorm = Math.sqrt(adj.indices.reduce(function(s, i) { return s + newHub[i] * newHub[i]; }, 0)) || 1;
      adj.indices.forEach(function(i) { newHub[i] /= hubNorm; });

      var diff = 0;
      adj.indices.forEach(function(i) { diff += Math.abs(newHub[i] - hub[i]) + Math.abs(newAuth[i] - auth[i]); });
      hub = newHub;
      auth = newAuth;
      if (diff < tol) break;
    }

    return { hub: hub, authority: auth };
  }

  // Min-max normalize a {index -> value} map to [0,1] — matches d3graph.py's
  // server-side _normalize_dict() exactly, which is applied uniformly to
  // ALL six stats there. Doing the same here (rather than the previous
  // divide-by-max, and rather than leaving degree/closeness/betweenness
  // unnormalized) keeps the color scale's [0, 0.5, 1] domain fully used
  // regardless of which stat is active or how the slider has filtered the
  // network — a stat whose raw scores are all clustered near 0 (e.g.
  // betweenness on a sparse graph) still stretches across the full range
  // instead of rendering as a single flat color.
  function normalizeMinMax(values, indices) {
    var vmin = Infinity, vmax = -Infinity;
    indices.forEach(function(i) {
      var v = values[i];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    });
    var range = vmax - vmin;
    var result = {};
    indices.forEach(function(i) {
      result[i] = (range < 1e-12) ? 0 : (values[i] - vmin) / range;
    });
    return result;
  }

  function computeDynamicStatValues(key, adj) {
    var raw;
    switch (key) {
      case 'node_degree_centrality':     raw = computeDegreeCentrality(adj); break;
      case 'node_closeness_centrality':  raw = computeClosenessCentrality(adj); break;
      case 'node_betweenness_centrality':raw = computeBetweennessCentrality(adj); break;
      case 'node_pagerank':              raw = computePageRank(adj); break;
      case 'node_hits_hub':              raw = computeHITS(adj).hub; break;
      case 'node_hits_authority':        raw = computeHITS(adj).authority; break;
      default: return null;
    }
    return normalizeMinMax(raw, adj.indices);
  }

  // Groups nodes into connected components using ONLY the edges currently
  // in `graph.links` — i.e. whatever survives the weight-threshold slider
  // right now. Re-run every time applyStatStyling() is called while
  // 'network_clustering' is the active stat, so the coloring always
  // reflects the live, filtered network rather than a one-time snapshot.
  // Cheap union-find over nodes/links; fine at interactive graph sizes.
  function computeNetworkClusters() {
    var parent = {};
    function find(x) {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
      }
      return x;
    }
    function union(a, b) {
      var ra = find(a), rb = find(b);
      if (ra !== rb) parent[ra] = rb;
    }

    graph.nodes.forEach(function(d) { parent[d.index] = d.index; });
    graph.links.forEach(function(l) {
      var s = (l.source && l.source.index !== undefined) ? l.source.index : l.source;
      var t = (l.target && l.target.index !== undefined) ? l.target.index : l.target;
      union(s, t);
    });

    // Re-number component roots to small, dense integers (0, 1, 2, ...) so
    // they map cleanly onto the categorical color scale.
    networkClusters = {};
    var rootToClusterId = {};
    var nextId = 0;
    graph.nodes.forEach(function(d) {
      var root = find(d.index);
      if (!(root in rootToClusterId)) rootToClusterId[root] = nextId++;
      networkClusters[d.index] = rootToClusterId[root];
    });
  }

  // Recolors AND resizes node shapes by the currently selected statistic
  // (reverting to each node's original color/size/opacity when none is
  // selected), and boosts edge width/opacity for readability against the
  // now-emphasized nodes. Handles circle/ellipse/path shapes generically,
  // same approach as the click-to-highlight scaling. Safe to call even if a
  // node is missing the stat (e.g. older data) — falls back to defaults.
  //
  // 'network_clustering' is handled as a variant of this same flow: it has
  // no single [0,1] value to size nodes by (cluster ids are categorical,
  // not a magnitude), so size stays at the default scale, but fill color
  // still comes from the stat — just from the categorical cluster palette
  // instead of statColorScale. Clusters are recomputed here every call, so
  // switching the weight-threshold slider (which rebuilds graph.links and
  // calls restart() -> applyStatStyling()) immediately updates node colors
  // to match the new connected-component structure.
  //
  // The other stats (PageRank, HITS, degree/closeness/betweenness
  // centrality) get the same live-recompute treatment: DYNAMIC_STAT_KEYS
  // are recalculated from the current graph.links before being read below,
  // so they also update immediately on every slider change.
  function applyStatStyling() {
    var isClustering = (currentStatKey === 'network_clustering');
    statHighlightActive = !!currentStatKey && !isClustering;

    if (isClustering) computeNetworkClusters();

    if (currentStatKey && DYNAMIC_STAT_KEYS.indexOf(currentStatKey) !== -1) {
      var adj = buildAdjacency();
      var values = computeDynamicStatValues(currentStatKey, adj);
      if (values) {
        graph.nodes.forEach(function(d) { d[currentStatKey] = values[d.index]; });
      }
    }

    node.select(".node-shape").each(function(d) {
      var el = d3.select(this);
      var tag = this.tagName.toLowerCase();
      var baseR = parseFloat(d.node_size) || 8;
      var v = statHighlightActive ? d[currentStatKey] : null;
      var hasVal = (typeof v === 'number' && !isNaN(v));
      // 1.2x–2.5x range: even low scorers get a visibility bump, high scorers stand out more.
      var scale = (statHighlightActive && hasVal) ? (1.2 + v * 1.3) : 1;
      // Clustering has no [0,1] magnitude to size by, but it does have a
      // categorical color per component — use that for fill instead of
      // falling back to the node's original color.
      var fillColor = isClustering
        ? clusterColorScale(networkClusters[d.index] % 20)
        : ((statHighlightActive && hasVal) ? statColorScale(v) : d.node_color);

      if (tag === 'circle') {
        el.attr('r', baseR * scale);
      } else if (tag === 'ellipse') {
        el.attr('rx', baseR * 1.6 * scale).attr('ry', baseR * scale);
      } else if (tag === 'path') {
        el.attr('d', shapePathD(d.node_marker, baseR * scale));
      }

      el.style('fill', fillColor)
        .style('opacity', statHighlightActive ? 1 : d.node_opacity)
        .style('stroke-width', statHighlightActive ? (parseFloat(d.node_size_edge) || 1) * 1.5 : d.node_size_edge);
    });

    // SVG-mode edges: bump width/opacity directly on the current selection.
    // Canvas-mode edges read statHighlightActive inside drawCanvasEdges().
    link.style("stroke-width", function(d) {
      var w = parseFloat(d.edge_width) || 1;
      return statHighlightActive ? w * EDGE_HIGHLIGHT_WIDTH_MULT : w;
    });
    link.style("opacity", function(d) {
      var o = (d.edge_opacity !== undefined && d.edge_opacity !== null) ? d.edge_opacity : 0.6;
      return statHighlightActive ? Math.min(1, o * EDGE_HIGHLIGHT_OPACITY_MULT) : o;
    });

    // Node label color: cluster color while clustering mode is active (so
    // the label visibly matches its node's cluster), otherwise each node's
    // own configured font color. Always set explicitly (rather than
    // clearing the inline style) so switching away from clustering cleanly
    // restores the original per-node color instead of falling through to
    // the CSS default.
    node.select("text").style("fill", function(d) {
      return isClustering ? clusterColorScale(networkClusters[d.index] % 20) : d.node_fontcolor;
    });

    if (useCanvasEdges) drawCanvasEdges();
  }

  var statRadios = document.querySelectorAll('input[name="statMetric"]');
  if (statRadios.length) {
    statRadios.forEach(function(radio) {
      radio.addEventListener('change', function() {
        currentStatKey = (this.value === 'none') ? null : this.value;
        applyStatStyling();
      });
    });
  }

  // Arranges nodes evenly around a circle. Order follows each node's stable
  // .index (assignment order), not the current force-settled position, so
  // the arrangement is deterministic and doesn't depend on prior layout.
  function layoutCircular() {
    var nodesData = graph.nodes;
    var n = nodesData.length;
    if (!n) return;
    var cx = width / 2, cy = height / 2;
    var r = Math.min(width, height) * 0.4;
    nodesData.forEach(function(d, i) {
      var angle = (i / n) * 2 * Math.PI;
      d.x = d.px = cx + r * Math.cos(angle);
      d.y = d.py = cy + r * Math.sin(angle);
    });
  }

  // Arranges nodes in a roughly-square grid.
  function layoutGrid() {
    var nodesData = graph.nodes;
    var n = nodesData.length;
    if (!n) return;
    var cols = Math.max(1, Math.ceil(Math.sqrt(n)));
    var rows = Math.max(1, Math.ceil(n / cols));
    var spacingX = width / (cols + 1);
    var spacingY = height / (rows + 1);
    nodesData.forEach(function(d, i) {
      var col = i % cols;
      var row = Math.floor(i / cols);
      d.x = d.px = spacingX * (col + 1);
      d.y = d.py = spacingY * (row + 1);
    });
  }

  // Switches between the physics-simulated force layout (default) and
  // static, one-shot arrangements. Static layouts stop the simulation so
  // they don't get pulled back out of shape by ongoing forces; switching
  // back to 'force' just resumes the simulation from wherever nodes
  // currently are.
  function setLayout(layoutName) {
    currentLayout = layoutName;

    if (layoutName === 'force') {
      tickCount = 0;
      force.resume();
      return;
    }

    force.stop();
    if (layoutName === 'circular') {
      layoutCircular();
    } else if (layoutName === 'grid') {
      layoutGrid();
    }
    updatePositions();
  }

  var layoutRadios = document.querySelectorAll('input[name="layoutMode"]');
  if (layoutRadios.length) {
    layoutRadios.forEach(function(radio) {
      radio.addEventListener('change', function() {
        setLayout(this.value);
      });
    });
  }

  // ---- PHYSICS PANEL (charge / collision / link tension sliders) ----
  // Snapshot the values the graph was generated with, so Reset can restore
  // them exactly — not some hardcoded JS default, but whatever was actually
  // passed into show().
  var DEFAULT_CHARGE = config.charge;
  var DEFAULT_COLLISION = config.collision;
  var DEFAULT_LINK_TENSION = (config.link_tension !== undefined && config.link_tension !== null) ? config.link_tension : 1;

  // Only reheats the simulation if the force layout is actually the one
  // active — if a static layout (circular/grid) is selected, the new value
  // is stored and takes effect next time the user switches back to it,
  // rather than yanking them out of the layout they chose.
  function reheatIfForceActive() {
    if (currentLayout === 'force') {
      tickCount = 0;
      force.start();
    }
  }

  function setCharge(value) {
    config.charge = value;
    force.charge(value);
    reheatIfForceActive();
  }
  function setCollision(value) {
    // Read live every tick via node.each(collide(config.collision)) — no
    // force.* setter call needed, just update the value it reads.
    config.collision = value;
    reheatIfForceActive();
  }
  function setLinkTension(value) {
    config.link_tension = value;
    force.linkStrength(value);
    reheatIfForceActive();
  }

  function wirePhysicsSlider(sliderId, valueId, onChange) {
    var slider = document.getElementById(sliderId);
    var valueLabel = document.getElementById(valueId);
    if (!slider) return;
    slider.addEventListener('input', function() {
      var v = parseFloat(this.value);
      if (valueLabel) valueLabel.textContent = v;
      onChange(v);
    });
  }
  wirePhysicsSlider('chargeSlider', 'chargeValue', setCharge);
  wirePhysicsSlider('collisionSlider', 'collisionValue', setCollision);
  wirePhysicsSlider('linkTensionSlider', 'linkTensionValue', setLinkTension);

  var physicsResetBtn = document.getElementById('physicsResetButton');
  if (physicsResetBtn) {
    physicsResetBtn.addEventListener('click', function() {
      setCharge(DEFAULT_CHARGE);
      setCollision(DEFAULT_COLLISION);
      setLinkTension(DEFAULT_LINK_TENSION);

      var chargeSlider = document.getElementById('chargeSlider');
      var collisionSlider = document.getElementById('collisionSlider');
      var tensionSlider = document.getElementById('linkTensionSlider');
      if (chargeSlider) chargeSlider.value = DEFAULT_CHARGE;
      if (collisionSlider) collisionSlider.value = DEFAULT_COLLISION;
      if (tensionSlider) tensionSlider.value = DEFAULT_LINK_TENSION;

      var chargeVal = document.getElementById('chargeValue');
      var collisionVal = document.getElementById('collisionValue');
      var tensionVal = document.getElementById('linkTensionValue');
      if (chargeVal) chargeVal.textContent = DEFAULT_CHARGE;
      if (collisionVal) collisionVal.textContent = DEFAULT_COLLISION;
      if (tensionVal) tensionVal.textContent = DEFAULT_LINK_TENSION;
    });
  }

  // ---- DENSITY PANEL (grid size / blur / opacity sliders) ----
  // Snapshot the values the graph was generated with, so Reset restores
  // exactly those, not a hardcoded JS default.
  var DEFAULT_DENSITY_GRID_SIZE = densityGridSize;
  var DEFAULT_DENSITY_BLUR = densityBlur;
  var DEFAULT_DENSITY_OPACITY = densityOpacity;

  // Only visually matters while the density layer is toggled on, but the
  // values are stored either way (so turning the layer on later reflects
  // whatever the sliders were last set to).
  function setDensityGridSize(value) {
    densityGridSize = value;
    if (densityVisible) drawDensityLayer();
  }
  function setDensityBlur(value) {
    densityBlur = value;
    if (densityVisible) drawDensityLayer();
  }
  function setDensityOpacity(value) {
    densityOpacity = value;
    if (densityVisible) drawDensityLayer();
  }
  wirePhysicsSlider('densityGridSizeSlider', 'densityGridSizeValue', setDensityGridSize);
  wirePhysicsSlider('densityBlurSlider', 'densityBlurValue', setDensityBlur);
  wirePhysicsSlider('densityOpacitySlider', 'densityOpacityValue', setDensityOpacity);

  var densityResetBtn = document.getElementById('densityResetButton');
  if (densityResetBtn) {
    densityResetBtn.addEventListener('click', function() {
      setDensityGridSize(DEFAULT_DENSITY_GRID_SIZE);
      setDensityBlur(DEFAULT_DENSITY_BLUR);
      setDensityOpacity(DEFAULT_DENSITY_OPACITY);

      var gridSlider = document.getElementById('densityGridSizeSlider');
      var blurSlider = document.getElementById('densityBlurSlider');
      var opacitySlider = document.getElementById('densityOpacitySlider');
      if (gridSlider) gridSlider.value = DEFAULT_DENSITY_GRID_SIZE;
      if (blurSlider) blurSlider.value = DEFAULT_DENSITY_BLUR;
      if (opacitySlider) opacitySlider.value = DEFAULT_DENSITY_OPACITY;

      var gridVal = document.getElementById('densityGridSizeValue');
      var blurVal = document.getElementById('densityBlurValue');
      var opacityVal = document.getElementById('densityOpacityValue');
      if (gridVal) gridVal.textContent = DEFAULT_DENSITY_GRID_SIZE;
      if (blurVal) blurVal.textContent = DEFAULT_DENSITY_BLUR;
      if (opacityVal) opacityVal.textContent = DEFAULT_DENSITY_OPACITY;
    });
  }

  // Exports the currently filtered nodes/edges (respecting the weight and
  // component sliders — whatever's "in scope" right now, not necessarily
  // what's on screen if edges are toggled off) together with their
  // precomputed stats, as a single JSON file.
  var exportDataBtn = document.getElementById('exportDataButton');
  if (exportDataBtn) {
    exportDataBtn.addEventListener('click', function() {
      var nodesOut = graph.nodes.map(function(d) {
        return {
          id: d.index,
          name: d.node_name,
          pagerank: d.node_pagerank,
          hits_hub: d.node_hits_hub,
          hits_authority: d.node_hits_authority,
          degree_centrality: d.node_degree_centrality,
          closeness_centrality: d.node_closeness_centrality,
          betweenness_centrality: d.node_betweenness_centrality
        };
      });
      var edgesOut = graph.links.map(function(d) {
        return {
          source: (d.source && d.source.node_name !== undefined) ? d.source.node_name : d.source,
          target: (d.target && d.target.node_name !== undefined) ? d.target.node_name : d.target,
          weight: d.edge_weight
        };
      });
      var payload = { nodes: nodesOut, edges: edgesOut };
      var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = '{{ title }}_export.json';
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  //Restart the visualisation after any node and link changes
  function restart() {

    renderLinks();

    node = node.data(graph.nodes);
    node.enter().insert("circle", ".cursor").attr("class", "node").attr("r", 5).call(force.drag);

    if (currentLayout === 'force') {
      tickCount = 0;
      force.start();
    } else {
      // Static layout active: re-apply it (new/changed nodes need positions
      // too) instead of reheating the physics simulation.
      if (currentLayout === 'circular') layoutCircular();
      else if (currentLayout === 'grid') layoutGrid();
      updatePositions();
    }
    applyStatStyling();
  }

    function resizeGraph() {
    
        if (config.width === window.innerWidth) {
            width = window.innerWidth;
        }
    
        if (config.height === null) {
            var header = document.querySelector(".header-row");
            var headerHeight = header ? header.offsetHeight : 0;
            height = window.innerHeight - headerHeight;
        }
    
        container
            .style("width", width + "px")
            .style("height", height + "px");
    
        container.select("svg")
            .attr("width", width)
            .attr("height", height);
    
        canvasEl.width = width;
        canvasEl.height = height;
    
        densityCanvasEl.width = width;
        densityCanvasEl.height = height;
    
        force.size([width, height]).resume();
    
        drawCanvasEdges();
        drawDensityLayer();
    }
    
    window.addEventListener("resize", resizeGraph);

}